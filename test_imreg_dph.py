import numpy as np
from imreg_dph import AffineTransform
from nose.tools import *


def test_translation():
    """make sure tranlation works"""
    # AffineTransform Tests
    af1 = AffineTransform(translation=(1, 2))
    af2 = AffineTransform(translation=(5, 3))
    af3 = af1 @ af2
    assert np.array_equal(af3.translation, (6, 5))
    assert af3 == af2 @ af1


def test_rotation():
    """Test that rotation works"""
    af1 = AffineTransform(rotation=2)
    af2 = AffineTransform(rotation=1)
    af3 = af1 @ af2
    assert af3.rotation == 3
    assert af3 == af2 @ af1
