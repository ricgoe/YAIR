import pytest
import cv2 as cv
import numpy as np
import faiss
from features import SiftVecCalculator

@pytest.fixture(scope="module")
def dummy_kmeans():
    """
    Create a dummy FAISS index with random centroids for testing.
    """
    vector_length = 10
    dim = 128
    centroids = np.random.rand(vector_length, dim).astype(np.float32)
    
    
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    ids = np.arange(vector_length).astype(np.int64)
    index.add_with_ids(centroids, ids)
    return index

@pytest.fixture
def sift_calc(dummy_kmeans):
    return SiftVecCalculator(model=dummy_kmeans, vector_length=512)

def test_sift_vec_shape(sift_calc):
    """
    Test that a valid grayscale image produces the correct SIFT vector shape and type.
    """
    # random grayscale
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    vec = sift_calc.gen_sift_vec(img)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 512), f"Expected shape (1, 512), got {vec.shape}"
    assert vec.dtype == np.float32