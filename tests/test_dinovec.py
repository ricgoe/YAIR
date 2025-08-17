import pytest
import numpy as np
from features import DINOVecCalculator


@pytest.fixture(scope="module")
def dino_calc():
    """
    Initialize the DINO vector calculator
    """
    return DINOVecCalculator()

def test_dino_vector_shape_with_dummy_image(dino_calc):
    """
    Test that the generated vector from a dummy RGB image has the correct shape and dtype.
    """
    # dummy RGB image
    dummy_img = np.ones((256, 256, 3), dtype=np.uint8) * 127

    vec = dino_calc.gen_dino_vec(dummy_img)

    assert isinstance(vec, np.ndarray), "Output is not a NumPy array"
    assert vec.shape == (1, 384), f"Expected shape (1, 384), got {vec.shape}"
    assert vec.dtype == np.float32 or vec.dtype == np.float64, f"Unexpected dtype: {vec.dtype}"

def test_dino_vector_consistency_with_identical_input(dino_calc):
    """
    Test that the model has deterministic behavior.
    """
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    vec1 = dino_calc.gen_dino_vec(dummy_img)
    vec2 = dino_calc.gen_dino_vec(dummy_img)

    np.testing.assert_allclose(vec1, vec2, rtol=1e-5, atol=1e-6)