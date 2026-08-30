import numpy as np

from retrieval.mmr import MMRRetriever


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert MMRRetriever.cosine(a, b) == 1.0
    assert MMRRetriever.cosine(a, c) == 0.0
