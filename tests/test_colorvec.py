import pytest
import numpy as np
import cv2 as cv
from features import ColorVecCalculator

@pytest.fixture
def color_calc():
    return ColorVecCalculator(length=26)

def test_color_vector_output_shape(color_calc):
    """
    Test that the color vector has the correct shape and type.
    """
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    vec = color_calc.gen_color_vec(dummy_img)

    assert isinstance(vec, np.ndarray), "Output is not a NumPy array"
    assert vec.shape == (1, 26), f"Expected shape (1, 26), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected dtype float32, got {vec.dtype}" # for faiss

def test_quantization_range(color_calc):
    """
    Test that quantized values are within expected bin ranges.
    """
    dummy_hls = np.stack([
        np.full((100, 100), 90),   # hue midrange
        np.full((100, 100), 130),  # lightness > 20
        np.full((100, 100), 200)   # saturation > 50
    ], axis=-1).astype(np.uint8)

    quantized = color_calc.quantize_channels(dummy_hls)

    assert quantized.shape[-1] == 3
    for i in range(3):
        assert quantized[..., i].max() < color_calc.hls_bins[i], f"Value exceeds bin range for channel {i}"
        assert quantized[..., i].min() >= 0