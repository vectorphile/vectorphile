"""N-dimensional image processing — wraps scipy.ndimage."""
from __future__ import annotations
import numpy as np

try:
    import scipy.ndimage as _nd

    # Filtering
    def gaussian_filter(x, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
        return _nd.gaussian_filter(x, sigma, order=order, output=output, mode=mode, cval=cval, truncate=truncate)

    def gaussian_filter1d(x, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
        return _nd.gaussian_filter1d(x, sigma, axis=axis, order=order, output=output, mode=mode, cval=cval, truncate=truncate)

    def uniform_filter(x, size=3, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.uniform_filter(x, size=size, output=output, mode=mode, cval=cval, origin=origin)

    def uniform_filter1d(x, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.uniform_filter1d(x, size, axis=axis, output=output, mode=mode, cval=cval, origin=origin)

    def median_filter(x, size=3, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.median_filter(x, size=size, footprint=footprint, output=output, mode=mode, cval=cval, origin=origin)

    def maximum_filter(x, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.maximum_filter(x, size=size, footprint=footprint, output=output, mode=mode, cval=cval, origin=origin)

    def minimum_filter(x, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.minimum_filter(x, size=size, footprint=footprint, output=output, mode=mode, cval=cval, origin=origin)

    def laplace(x, output=None, mode='reflect', cval=0.0):
        return _nd.laplace(x, output=output, mode=mode, cval=cval)

    def gaussian_laplace(x, sigma, output=None, mode='reflect', cval=0.0, **kwargs):
        return _nd.gaussian_laplace(x, sigma, output=output, mode=mode, cval=cval, **kwargs)

    def sobel(x, axis=-1, output=None, mode='reflect', cval=0.0):
        return _nd.sobel(x, axis=axis, output=output, mode=mode, cval=cval)

    def prewitt(x, axis=-1, output=None, mode='reflect', cval=0.0):
        return _nd.prewitt(x, axis=axis, output=output, mode=mode, cval=cval)

    def generic_gradient_magnitude(x, derivative, output=None, mode='reflect', cval=0.0, extra_arguments=(), extra_keywords={}):
        return _nd.generic_gradient_magnitude(x, derivative, output=output, mode=mode, cval=cval,
                                               extra_arguments=extra_arguments, extra_keywords=extra_keywords)

    def convolve(x, weights, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.convolve(x, weights, output=output, mode=mode, cval=cval, origin=origin)

    def correlate(x, weights, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.correlate(x, weights, output=output, mode=mode, cval=cval, origin=origin)

    # Morphology
    def binary_dilation(x, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
        return _nd.binary_dilation(x, structure=structure, iterations=iterations, mask=mask, output=output, border_value=border_value, origin=origin, brute_force=brute_force)

    def binary_erosion(x, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
        return _nd.binary_erosion(x, structure=structure, iterations=iterations, mask=mask, output=output, border_value=border_value, origin=origin, brute_force=brute_force)

    def binary_opening(x, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False):
        return _nd.binary_opening(x, structure=structure, iterations=iterations, output=output, origin=origin, mask=mask, border_value=border_value, brute_force=brute_force)

    def binary_closing(x, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False):
        return _nd.binary_closing(x, structure=structure, iterations=iterations, output=output, origin=origin, mask=mask, border_value=border_value, brute_force=brute_force)

    def binary_fill_holes(x, structure=None, output=None, origin=0):
        return _nd.binary_fill_holes(x, structure=structure, output=output, origin=origin)

    def grey_dilation(x, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.grey_dilation(x, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval, origin=origin)

    def grey_erosion(x, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.grey_erosion(x, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval, origin=origin)

    def grey_opening(x, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.grey_opening(x, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval, origin=origin)

    def grey_closing(x, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.grey_closing(x, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval, origin=origin)

    def morphological_gradient(x, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
        return _nd.morphological_gradient(x, size=size, footprint=footprint, structure=structure, output=output, mode=mode, cval=cval, origin=origin)

    # Measurements
    def label(x, structure=None, output=None):
        return _nd.label(x, structure=structure, output=output)

    def find_objects(x, max_label=0):
        return _nd.find_objects(x, max_label=max_label)

    def center_of_mass(x, labels=None, index=None):
        return _nd.center_of_mass(x, labels=labels, index=index)

    def ndimage_sum(x, labels=None, index=None):
        return _nd.sum(x, labels=labels, index=index)

    def mean(x, labels=None, index=None):
        return _nd.mean(x, labels=labels, index=index)

    def variance(x, labels=None, index=None):
        return _nd.variance(x, labels=labels, index=index)

    def standard_deviation(x, labels=None, index=None):
        return _nd.standard_deviation(x, labels=labels, index=index)

    def maximum(x, labels=None, index=None):
        return _nd.maximum(x, labels=labels, index=index)

    def minimum(x, labels=None, index=None):
        return _nd.minimum(x, labels=labels, index=index)

    def maximum_position(x, labels=None, index=None):
        return _nd.maximum_position(x, labels=labels, index=index)

    def minimum_position(x, labels=None, index=None):
        return _nd.minimum_position(x, labels=labels, index=index)

    def histogram(x, min, max, bins, labels=None, index=None):
        return _nd.histogram(x, min, max, bins, labels=labels, index=index)

    # Interpolation / Geometric transforms
    def zoom(x, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True, **kwargs):
        return _nd.zoom(x, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def rotate(x, angle, axes=(1, 0), reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
        return _nd.rotate(x, angle, axes=axes, reshape=reshape, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def affine_transform(x, matrix, offset=0.0, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
        return _nd.affine_transform(x, matrix, offset=offset, output_shape=output_shape, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def map_coordinates(x, coordinates, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
        return _nd.map_coordinates(x, coordinates, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def shift(x, shift, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
        return _nd.shift(x, shift, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def distance_transform_edt(x, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None):
        return _nd.distance_transform_edt(x, sampling=sampling, return_distances=return_distances, return_indices=return_indices, distances=distances, indices=indices)

    def distance_transform_bf(x, metric='euclidean', sampling=None, return_distances=True, return_indices=False, distances=None, indices=None):
        return _nd.distance_transform_bf(x, metric=metric, sampling=sampling, return_distances=return_distances, return_indices=return_indices, distances=distances, indices=indices)

    def generate_binary_structure(rank, connectivity):
        return _nd.generate_binary_structure(rank, connectivity)

    def iterate_structure(structure, iterations, origin=None):
        return _nd.iterate_structure(structure, iterations, origin=origin)

except ImportError:
    def _no_scipy(*args, **kwargs):
        raise ImportError("scipy is required for vp.ndimage. Install with: pip install scipy")
    gaussian_filter = gaussian_filter1d = uniform_filter = uniform_filter1d = _no_scipy
    median_filter = maximum_filter = minimum_filter = laplace = gaussian_laplace = _no_scipy
    sobel = prewitt = generic_gradient_magnitude = convolve = correlate = _no_scipy
    binary_dilation = binary_erosion = binary_opening = binary_closing = _no_scipy
    binary_fill_holes = grey_dilation = grey_erosion = grey_opening = grey_closing = _no_scipy
    morphological_gradient = label = find_objects = center_of_mass = _no_scipy
    ndimage_sum = mean = variance = standard_deviation = maximum = minimum = _no_scipy
    maximum_position = minimum_position = histogram = zoom = rotate = _no_scipy
    affine_transform = map_coordinates = shift = distance_transform_edt = _no_scipy
    distance_transform_bf = generate_binary_structure = iterate_structure = _no_scipy
