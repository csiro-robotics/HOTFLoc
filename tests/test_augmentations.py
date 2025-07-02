"""
Test validity of augmentations.
"""
import pytest
import torch

from dataset.augmentation import Normalize

ATOL = 1e-6
dummy_points = torch.randn(1000,3) * 10.0

### Testing Normalize ###
def test_unnormalize():
    norm = Normalize(return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    assert torch.all(torch.isclose(norm.unnormalize(dummy_points_norm, shift_and_scale), dummy_points, atol=ATOL))
    
def test_unnormalize_unit_sphere():
    norm = Normalize(unit_sphere_norm=True, return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    assert torch.all(torch.isclose(norm.unnormalize(dummy_points_norm, shift_and_scale), dummy_points, atol=ATOL))
    
def test_unnormalize_large_scale_factor():
    norm = Normalize(scale_factor=100, return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    assert torch.all(torch.isclose(norm.unnormalize(dummy_points_norm, shift_and_scale), dummy_points, atol=ATOL))
    
def test_unnormalize_no_centering():
    norm = Normalize(zero_mean=False, return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    assert torch.all(torch.isclose(norm.unnormalize(dummy_points_norm, shift_and_scale), dummy_points, atol=ATOL))

def test_unnormalize_negative_scale():
    norm = Normalize(return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    shift_and_scale = (shift_and_scale[0], -100.)
    with pytest.raises(ValueError, match='Invalid scaling factor'):
        norm.unnormalize(dummy_points_norm, shift_and_scale)

def test_unnormalize_zero_scale():
    norm = Normalize(return_shift_and_scale=True)
    dummy_points_norm, shift_and_scale = norm(dummy_points)
    shift_and_scale = (shift_and_scale[0], 0.)
    with pytest.raises(ValueError, match='Invalid scaling factor'):
        norm.unnormalize(dummy_points_norm, shift_and_scale)

if __name__ == '__main__':
    test_unnormalize()
    test_unnormalize_unit_sphere()
    test_unnormalize_large_scale_factor()
    test_unnormalize_no_centering()
    test_unnormalize_negative_scale()
    test_unnormalize_zero_scale()