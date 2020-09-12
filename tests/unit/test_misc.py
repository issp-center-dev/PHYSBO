import sys

import pytest
import numpy as np

combo = pytest.importorskip("combo")


@pytest.fixture
def X():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )

# @pytest.mark.parametrize("index", [2, [0, 1]])
def test_centering(X):
    dim = X.shape[1]
    centered = combo.misc.centering(X)
    assert np.array_equal(centered.mean(axis=0), np.zeros(dim))
    assert np.array_equal(centered.std(axis=0), np.ones(dim))
