from __future__ import annotations

import numpy as np

from eta import linear_eta as production_linear_eta
from training.rival_training.wisp_contract import linear_eta as contract_linear_eta


def test_contract_linear_eta_matches_production_on_randomized_domain() -> None:
    rng = np.random.default_rng(20260823)
    for velocity, distance, boost_duration in zip(
        rng.uniform(-500.0, 2299.0, 2048),
        rng.uniform(25.0, 12000.0, 2048),
        rng.uniform(0.0, 3.1, 2048),
        strict=True,
    ):
        expected = production_linear_eta(
            float(velocity), float(distance), float(boost_duration)
        )
        actual = contract_linear_eta(
            float(velocity), float(distance), float(boost_duration)
        )
        assert actual == expected
