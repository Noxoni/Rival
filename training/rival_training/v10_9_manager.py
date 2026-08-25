"""rlgym-ppo manager that records AR state but sends only RivalActionV1."""

from __future__ import annotations

import numpy as np
import torch

from rlgym_ppo.batched_agents import BatchedAgentManager

from .v10_9_actions import ANALOG_DIM, pack_rollout_actions


class RivalAR1BatchedAgentManager(BatchedAgentManager):
    """Keep one AR epsilon state per environment learner trajectory."""

    def init_processes(self, *args, **kwargs):
        shapes = super().init_processes(*args, **kwargs)
        self.ar_epsilon: list[np.ndarray | None] = [None] * len(self.processes)
        self.ar_initial: list[np.ndarray | None] = [None] * len(self.processes)
        self.ar_reset_count = 0
        self.ar_transition_count = 0
        return shapes

    def _context(self, proc_id: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
        epsilon = self.ar_epsilon[proc_id]
        initial = self.ar_initial[proc_id]
        if epsilon is None or epsilon.shape != (rows, ANALOG_DIM):
            epsilon = np.zeros((rows, ANALOG_DIM), dtype=np.float32)
            initial = np.ones((rows, 1), dtype=np.float32)
            self.ar_epsilon[proc_id] = epsilon
            self.ar_initial[proc_id] = initial
            self.ar_reset_count += rows
        assert initial is not None
        return epsilon, initial

    @torch.no_grad()
    def _send_actions(self):
        if not self.current_pids:
            return
        dimensions = []
        observations = []
        previous_rows = []
        initial_rows = []
        pids = []
        for proc_id in self.current_pids:
            observation = self.current_obs[proc_id]
            if observation is None:
                continue
            rows = int(observation.shape[0])
            previous, initial = self._context(proc_id, rows)
            observations.append(observation)
            previous_rows.append(previous.copy())
            initial_rows.append(initial.copy())
            dimensions.append(rows)
            pids.append(proc_id)
        if not dimensions:
            return
        inference_batch = np.concatenate(observations, axis=0)
        previous_batch = np.concatenate(previous_rows, axis=0)
        initial_batch = np.concatenate(initial_rows, axis=0)
        physical, log_probabilities, epsilon = self.policy.sample_with_context(
            inference_batch, previous_batch, initial_batch
        )
        physical_np = physical.numpy().astype(np.float32)
        epsilon_np = epsilon.numpy().astype(np.float32)
        step = 0
        for proc_id, rows in zip(pids, dimensions):
            process, parent_end, child_endpoint, shm_view = self.processes[proc_id]
            stop = step + rows
            physical_slice = physical_np[step:stop]
            previous_slice = torch.as_tensor(previous_batch[step:stop])
            initial_slice = torch.as_tensor(initial_batch[step:stop])
            recorded = pack_rollout_actions(
                torch.as_tensor(physical_slice), previous_slice, initial_slice
            ).numpy()
            parent_end.sendto(
                self.packed_header + physical_slice.tobytes(), child_endpoint
            )
            self.trajectory_map[proc_id].action = recorded
            self.trajectory_map[proc_id].log_prob = log_probabilities[step:stop]
            self.trajectory_map[proc_id].state = inference_batch[step:stop]
            self.ar_epsilon[proc_id] = epsilon_np[step:stop].copy()
            self.ar_initial[proc_id] = np.zeros((rows, 1), dtype=np.float32)
            self.ar_transition_count += rows
            step = stop
        self.current_pids = []

    def _collect_response(self, proc_id, *args, **kwargs):
        collected = super()._collect_response(proc_id, *args, **kwargs)
        trajectory = self.trajectory_map[proc_id]
        if bool(trajectory.done) or bool(trajectory.truncated):
            rows = len(trajectory.next_state)
            self.ar_epsilon[proc_id] = np.zeros(
                (rows, ANALOG_DIM), dtype=np.float32
            )
            self.ar_initial[proc_id] = np.ones((rows, 1), dtype=np.float32)
            self.ar_reset_count += rows
        return collected

    def ar_state_diagnostics(self) -> dict[str, int]:
        return {
            "reset_trajectory_rows": int(self.ar_reset_count),
            "sampled_transition_rows": int(self.ar_transition_count),
            "active_state_slots": sum(row is not None for row in self.ar_epsilon),
        }
