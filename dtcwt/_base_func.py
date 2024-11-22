from __future__ import absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets.utils import symm_pad_1d as symm_pad

def as_column_vector(v):
    """Return *v* as a column vector with shape (N,1).

    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v
    
def _as_col_vector(v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v

def prep_filt(h, c, transpose=False):
    """ Prepares an array to be of the correct format for pytorch.
    Can also specify whether to make it a row filter (set tranpose=True)"""
    h = _as_col_vector(h)[::-1]
    h = h[None, None, :]
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0,1,3,2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.get_default_dtype())

def first_filter_1d(X, h, mode='symmetric', axis=-1):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1, device=X.device)
    if len(X.shape) == 4:
        h = h.swapaxes(-2, axis).contiguous()
        if mode == 'symmetric':
            xe = torch.from_numpy(symm_pad(X.shape[axis], h.shape[axis] // 2))
            X1 = F.conv2d(torch.index_select(X, axis, xe), h.repeat(X.shape[1],1,1,1), groups=X.shape[1])
        else:
            raise NotImplementedError()
        
    elif len(X.shape) == 5:
        h = h.unsqueeze(0).swapaxes(-2, axis).contiguous()
        if mode == 'symmetric':
            xe = torch.from_numpy(symm_pad(X.shape[axis], h.shape[axis] // 2))
            X1 = F.conv3d(torch.index_select(X, axis, xe), h.repeat(X.shape[1],1,1,1,1), groups=X.shape[1])
        else:
            raise NotImplementedError()

    else:
        raise NotImplemented('Input tensor must be 2D or 3D')
    
    return X1
    
def plus_filter_1d(X, ha, hb, highpass=False, mode='symmetric', axis=-1):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1, device=X.device)
    if len(X.shape) == 4:
        ha = ha.swapaxes(-2, axis).contiguous()
        hb = hb.swapaxes(-2, axis).contiguous()
        shapes = list(X.shape)
        shapes[axis] = shapes[axis] // 2
        batch, ch, r, c = tuple(shapes)
        if X.shape[axis] % 4 != 0:
            raise ValueError('No. of rows in X must be a multiple of 4\n' +
                             'X was {}'.format(X.shape))
        if mode == 'symmetric':
            xe = torch.from_numpy(symm_pad(X.shape[axis], ha.shape[axis]))
            X = torch.cat([torch.index_select(X, axis, xe[2::2]), torch.index_select(X, axis, xe[3::2])], dim=1) # cat on channel dimension
            h = torch.cat([ha.repeat(ch,1,1,1), hb.repeat(ch,1,1,1)], dim=0) # cat on Number dimension
            stride = [1,1]
            stride[axis] += 1
            X1 = F.conv2d(X, h, stride=stride, groups=ch*2)
            if highpass:
                Y = torch.stack((X1[:, ch:], X1[:, :ch]), dim=axis).view(batch, ch, r, c)
            else:
                Y = torch.stack((X1[:, :ch], X1[:, ch:]), dim=axis).view(batch, ch, r, c)
        else: # mode
            raise NotImplementedError()
        
    elif len(X.shape) == 5:
        ha = ha.unsqueeze(0).swapaxes(-2, axis).contiguous()
        hb = hb.unsqueeze(0).swapaxes(-2, axis).contiguous()
        shapes = list(X.shape)
        shapes[axis] = shapes[axis] // 2
        batch, ch, d, r, c = shapes
        if mode == 'symmetric':
            xe = torch.from_numpy(symm_pad(X.shape[axis], ha.shape[axis]))
            X = torch.cat([torch.index_select(X, axis, xe[2::2]), torch.index_select(X, axis, xe[3::2])], dim=1) # cat on channel dimension
            h = torch.cat([ha.repeat(ch,1,1,1,1), hb.repeat(ch,1,1,1,1)], dim=0) # cat on Number dimension
            stride = [1,1,1]
            stride[axis] += 1
            X1 = F.conv3d(X, h, stride=stride, groups=ch*2)
            if highpass:
                Y = torch.stack((X1[:, ch:], X1[:, :ch]), dim=axis).view(batch, ch, d, r, c)
            else:
                Y = torch.stack((X1[:, :ch], X1[:, ch:]), dim=axis).view(batch, ch, d, r, c)
        else:
            raise NotImplementedError()
    else:
        raise NotImplemented('Input tensor must be 2D or 3D')
    
    return Y

#TODO: Version 1.5
def colifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    m = ha.shape[2]
    m2 = m // 2
    hao = ha[:,:,1::2]
    hae = ha[:,:,::2]
    hbo = hb[:,:,1::2]
    hbe = hb[:,:,::2]
    batch, ch, r, c = X.shape
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2.\n' +
                         'X was {}'.format(X.shape))
    xe = symm_pad(r, m2)

    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:,:,xe[1:-2:2]], X[:,:,xe[:-2:2]], X[:,:,xe[3::2]], X[:,:,xe[2::2]]), dim=1)
        else:
            X = torch.cat((X[:,:,xe[:-2:2]], X[:,:,xe[1:-2:2]], X[:,:,xe[2::2]], X[:,:,xe[3::2]]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]]), dim=1)
        else:
            X = torch.cat((X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]], X[:,:,xe[1:-1:2]], X[:,:,xe[2:-1:2]]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)), dim=0)

    X = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    X = torch.stack([X[:,:ch], X[:,ch:2*ch], X[:,2*ch:3*ch], X[:,3*ch:]], dim=3).view(batch, ch, r*2, c)

    return X

#TODO: Version 1.5
def rowifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=X.device)
    m = ha.shape[2]
    m2 = m // 2
    hao = ha[:,:,1::2]
    hae = ha[:,:,::2]
    hbo = hb[:,:,1::2]
    hbe = hb[:,:,::2]
    batch, ch, r, c = X.shape
    if c % 2 != 0:
        raise ValueError('No. of cols in X must be a multiple of 2.\n' +
                         'X was {}'.format(X.shape))
    xe = symm_pad(c, m2)

    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:,:,:,xe[1:-2:2]], X[:,:,:,xe[:-2:2]], X[:,:,:,xe[3::2]], X[:,:,:,xe[2::2]]), dim=1)
        else:
            X = torch.cat((X[:,:,:,xe[:-2:2]], X[:,:,:,xe[1:-2:2]], X[:,:,:,xe[2::2]], X[:,:,:,xe[3::2]]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]]), dim=1)
        else:
            X = torch.cat((X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]], X[:,:,:,xe[1:-1:2]], X[:,:,:,xe[2:-1:2]]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1),
                   h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)),
                  dim=0).reshape(4*ch, 1, 1, m2)

    X = F.conv2d(X, h, groups=4*ch)
    # Stack 4 tensors of shape [batch, ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    X = torch.stack([X[:,:ch], X[:,ch:2*ch], X[:,2*ch:3*ch], X[:,3*ch:]], dim=4).view(batch, ch, r, c*2)
    return X


def q2c(y, dim=-1):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    y = y/np.sqrt(2)
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    #  return torch.stack((a-d, b+c), dim=dim), torch.stack((a+d, b-c), dim=dim)
    z1 = a - d
    z2 = b + c
    z3 = a + d
    z4 = b - c
    return ((z1, z2), (z3, z4))


def q2c_3d(y, dim=-1):
    """
    Convert from quads in y to complex numbers in z. 3d-input have 4-orientation in total.
    """
    # Normalize the input
    y = y / np.sqrt(3)

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #      e--------f
    #     /|       /|
    #    / |      / |
    #   a--------b  |
    #   |  |     |  |
    #   |  g-----|--h
    #   | /      | /
    #   c--------d

    # Combine (a,b) and (d,c) to form two complex subimages.
    a, b = y[:, :, 0::2, 0::2, 0::2], y[:, :, 0::2, 0::2, 1::2]
    c, d = y[:, :, 0::2, 1::2, 0::2], y[:, :, 0::2, 1::2, 1::2]
    e, f = y[:, :, 1::2, 0::2, 0::2], y[:, :, 1::2, 0::2, 1::2]
    g, h = y[:, :, 1::2, 1::2, 0::2], y[:, :, 1::2, 1::2, 1::2]

    # Combine to form complex subimages
    # z1 = (a - h, b + g, c + f, d - e)
    # z2 = (a + h, b - g, c - f, d + e)
    # z3 = (a - h, b + g, c + f, d - e)
    # z4 = (a + h, b - g, c - f, d + e)

    # return ((z1, z2), (z3, z4))

    z1_real = a - h
    z1_imag = e + d
    z2_real = b - g
    z2_imag = f + c

    # Second sub-volume (opposite diagonal orientation)
    z3_real = a + h
    z3_imag = e - d
    z4_real = b + g
    z4_imag = f - c

    # Return as pairs of real and imaginary components
    return ((z1_real, z1_imag), (z2_real, z2_imag), (z3_real, z3_imag), (z4_real, z4_imag))


def c2q(w1, w2):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """
    w1r, w1i = w1
    w2r, w2i = w2

    x1 = w1r + w2r
    x2 = w1i + w2i
    x3 = w1i - w2i
    x4 = -w1r + w2r

    # Get the shape of the tensor excluding the real/imagniary part
    b, ch, r, c = w1r.shape

    # Create new empty tensor and fill it
    y = w1r.new_zeros((b, ch, r*2, c*2), requires_grad=w1r.requires_grad)
    y[:, :, ::2,::2] = x1
    y[:, :, ::2, 1::2] = x2
    y[:, :, 1::2, ::2] = x3
    y[:, :, 1::2, 1::2] = x4
    y /= np.sqrt(2)

    return y


#TODO: Version 1.5
def c2q_3d(w1, w2):
    pass