from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed

