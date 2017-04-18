import numpy as np
# three different registration packages
# not dft based
import cv2
# dft based
from skimage.feature import register_translation
from skimage.transform import warp
from skimage.transform import AffineTransform as AffineTransformBase
import pyfftw
pyfftw.interfaces.cache.enable()
from pyfftw.interfaces.numpy_fft import *
from dphutils import slice_maker


class AffineTransform(AffineTransformBase):
    """Only adding matrix multiply to previous class"""

    def __matmul__(self, other):
        newmat = self.params @ other.params
        return AffineTransform(matrix=newmat)

    def __eq__(self, other):
        return np.array_equal(self.params, other.params)

    def inverse(self):
        return AffineTransform(matrix=np.linalg.inv(self.params))


AffineTransform.__init__.__doc__ = AffineTransformBase.__init__.__doc__
AffineTransform.__doc__ = AffineTransformBase.__doc__


def localize_peak(data):
    """
    Small utility function to localize a peak center. Assumes passed data has
    peak at center and that data.shape is odd and symmetric. Then fits a
    parabola through each line passing through the center. This is optimized
    for FFT data which has a non-circularly symmetric shaped peaks.
    """
    # make sure passed data is symmetric along all dimensions
    if not len(set(data.shape)) == 1:
        print("data.shape = {}".format(data.shape))
        return 0, 0
    # pull center location
    center = data.shape[0] // 2
    # generate the fitting lines
    my_pat_fft_suby = data[:, center]
    my_pat_fft_subx = data[center, :]
    # fit along lines, consider the center to be 0
    x = np.arange(data.shape[0]) - center
    xfit = np.polyfit(x, my_pat_fft_subx, 2)
    yfit = np.polyfit(x, my_pat_fft_suby, 2)
    # calculate center of each parabola
    x0 = -xfit[1] / (2 * xfit[0])
    y0 = -yfit[1] / (2 * yfit[0])
    # NOTE: comments below may be useful later.
    # save fits as poly functions
    # ypoly = np.poly1d(yfit)
    # xpoly = np.poly1d(xfit)
    # peak_value = ypoly(y0) / ypoly(0) * xpoly(x0)
    # #
    # assert np.isclose(peak_value,
    #                   xpoly(x0) / xpoly(0) * ypoly(y0))
    # return center
    return y0, x0


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = np.empty((angles, radii), dtype=np.float64)
    theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
    #d = radii
    d = np.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (np.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
    x = (radius / shape[1] * shape[0]) * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    dt0, dt1 = localize_peak(ir[slice_maker(t0, t1, 3)])
    t0, t1 = t0 + dt0, t1 + dt1,
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return AffineTransform(translation=(-t1, -t0))


def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    - can fix with padding, non-square images can be handled either with padding or
        better yet compensating for uneven image size
    All image areas must have same scale, rotation, and shift.
    - tiling if necessary...
    Scale change must be less than 1.8.
    - why?
    No subpixel precision.
    - fit peak position or upsample as in (https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/register_translation.py)

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")
    shape_ratio = im0.shape[0] / im0.shape[1]
    # calculate fourier images of inputs
    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))
    # high pass filter fourier images
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
#     del h
    # convert images to logpolar coordinates.
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    # fourier transform again ?
    f0 = fft2(f0)
    f1 = fft2(f1)
    # calculate impulse response
    r0 = abs(f0) * abs(f1)
    ir_cmplx = ifft2((f0 * f1.conjugate()) / r0)
    ir = abs(ir_cmplx)
    # find max
    i0, i1 = np.array(np.unravel_index(np.argmax(ir), ir.shape))
    di0, di1 = localize_peak(ir[slice_maker(i0, i1, 5)])
    i0, i1 = i0 + di0, i1 + di1
    # calculate the angle
    angle = i0 / ir.shape[0]
    # and scale
    scale = log_base ** i1
    # if scale is too big, try complex conjugate of ir
    if scale > 1.8:
        ir = abs(ir_cmplx.conjugate())
        i0, i1 = np.array(np.unravel_index(np.argmax(ir), ir.shape))
        di0, di1 = localize_peak(ir[slice_maker(i0, i1, 5)])
        i0, i1 = i0 + di0, i1 + di1
        angle = -i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    # center the angle
    angle *= np.pi
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    # apply scale and rotation
    # first move center to 0, 0
    # center shift is reversed because of definition of AffineTransform
    center_shift = np.array(im1.shape)[::-1] // 2
    af = AffineTransform(translation=center_shift)
    # then apply scale and rotation
    af @= AffineTransform(scale=(scale, scale), rotation=angle)
    # move back to center of image
    af @= AffineTransform(translation=-center_shift)
    # apply transformation
    im2 = warp(im1, af)
    # now calculate translation
    af @= translation(im0, im2)
    im2 = warp(im1, af)

    return im2, af


def register_ECC(im0, im1, warp_mode=cv2.MOTION_AFFINE, num_iter=50, term_eps=1e-3):
    """
    # Specify the number of iterations.

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations;
    """
    # Find size of image1
    sz = im0.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter,  term_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    cc, warp_matrix = cv2.findTransformECC (im0, im1, warp_matrix, warp_mode, criteria)

    return AffineTransform2(matrix=np.vstack((warp_matrix, (0, 0, 1))))
