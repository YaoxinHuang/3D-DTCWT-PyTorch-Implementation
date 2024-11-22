import torch
from torch import tensor
from torch.autograd import Function
from pytorch_wavelets.dwt.lowlevel import int_to_mode
from dtcwt._base_func import first_filter_1d, plus_filter_1d
from dtcwt._base_func import colifilt, rowifilt, q2c, c2q, q2c_3d


def get_dimensions5(o_dim, ri_dim):
    """ Get the orientation, height and width dimensions after the real and
    imaginary parts have been popped off (5 dimensional tensor)."""
    o_dim = (o_dim % 6)
    ri_dim = (ri_dim % 6)

    if ri_dim < o_dim:
        o_dim -= 1

    if o_dim == 4:
        h_dim = 2
        w_dim = 3
    elif o_dim == 3:
        h_dim = 2
        w_dim = 4
    else:
        h_dim = 3
        w_dim = 4

    return o_dim, ri_dim, h_dim, w_dim


def get_dimensions6(o_dim, ri_dim):
    """ Get the orientation, real/imag, height and width dimensions
    for the full tensor (6 dimensions)."""
    # Calculate which dimension to put the real and imaginary parts and the
    # orientations. Also work out where the rows and columns in the original
    # image were
    o_dim = (o_dim % 6)
    ri_dim = (ri_dim % 6)

    if ri_dim < o_dim:
        o_dim -= 1

    if o_dim >= 3 and ri_dim >= 3:
        h_dim = 2
    elif o_dim >= 4 or ri_dim >= 4:
        h_dim = 3
    else:
        h_dim = 4

    if o_dim >= 4 and ri_dim >= 4:
        w_dim = 3
    elif o_dim >= 4 or ri_dim >= 4:
        w_dim = 4
    else:
        w_dim = 5

    return o_dim, ri_dim, h_dim, w_dim



def highs_to_orientations(lh, hl, hh, o_dim):
    (deg15r, deg15i), (deg165r, deg165i) = q2c(lh)
    (deg45r, deg45i), (deg135r, deg135i) = q2c(hh)
    (deg75r, deg75i), (deg105r, deg105i) = q2c(hl)

    # Convert real and imaginary to magnitude
    reals = torch.stack(
        [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=o_dim)
    imags = torch.stack(
        [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=o_dim)

    return reals, imags


def highs_to_orientations_3d(llh, lhl, lhh, hll, hlh, hhl, hhh, o_dim):
    """
    Convert high-frequency subbands to orientations for 3D data.
    The result has 28 real and 28 imaginary parts.
    TODO: Assure the orientation order is correct
    """
    # Apply q2c_3d to each subband
    # Each subband produces 4 real and 4 imaginary parts
    # q2c_3d's return: ((z1_real, z1_imag), (z2_real, z2_imag), (z3_real, z3_imag), (z4_real, z4_imag))
    (llh_r1, llh_i1), (llh_r2, llh_i2), (llh_r3, llh_i3), (llh_r4, llh_i4) = q2c_3d(llh)
    (lhl_r1, lhl_i1), (lhl_r2, lhl_i2), (lhl_r3, lhl_i3), (lhl_r4, lhl_i4) = q2c_3d(lhl)
    (lhh_r1, lhh_i1), (lhh_r2, lhh_i2), (lhh_r3, lhh_i3), (lhh_r4, lhh_i4) = q2c_3d(lhh)
    (hll_r1, hll_i1), (hll_r2, hll_i2), (hll_r3, hll_i3), (hll_r4, hll_i4) = q2c_3d(hll)
    (hlh_r1, hlh_i1), (hlh_r2, hlh_i2), (hlh_r3, hlh_i3), (hlh_r4, hlh_i4) = q2c_3d(hlh)
    (hhl_r1, hhl_i1), (hhl_r2, hhl_i2), (hhl_r3, hhl_i3), (hhl_r4, hhl_i4) = q2c_3d(hhl)
    (hhh_r1, hhh_i1), (hhh_r2, hhh_i2), (hhh_r3, hhh_i3), (hhh_r4, hhh_i4) = q2c_3d(hhh)

    # Stack all real parts (28 total)
    reals = torch.stack([
        llh_r1, llh_r2, llh_r3, llh_r4,
        lhl_r1, lhl_r2, lhl_r3, lhl_r4,
        lhh_r1, lhh_r2, lhh_r3, lhh_r4,
        hll_r1, hll_r2, hll_r3, hll_r4,
        hlh_r1, hlh_r2, hlh_r3, hlh_r4,
        hhl_r1, hhl_r2, hhl_r3, hhl_r4,
        hhh_r1, hhh_r2, hhh_r3, hhh_r4
    ], dim=o_dim)

    # Stack all imaginary parts (28 total)
    imags = torch.stack([
        llh_i1, llh_i2, llh_i3, llh_i4,
        lhl_i1, lhl_i2, lhl_i3, lhl_i4,
        lhh_i1, lhh_i2, lhh_i3, lhh_i4,
        hll_i1, hll_i2, hll_i3, hll_i4,
        hlh_i1, hlh_i2, hlh_i3, hlh_i4,
        hhl_i1, hhl_i2, hhl_i3, hhl_i4,
        hhh_i1, hhh_i2, hhh_i3, hhh_i4
    ], dim=o_dim)

    return reals, imags



def orientations_to_highs(reals, imags, o_dim):
    dev = reals.device
    horiz = torch.index_select(reals, o_dim, tensor([0, 5], device=dev))
    diag = torch.index_select(reals, o_dim, tensor([1, 4], device=dev))
    vertic = torch.index_select(reals, o_dim, tensor([2, 3], device=dev))
    deg15r, deg165r = torch.unbind(horiz, dim=o_dim)
    deg45r, deg135r = torch.unbind(diag, dim=o_dim)
    deg75r, deg105r = torch.unbind(vertic, dim=o_dim)
    dev = imags.device
    horiz = torch.index_select(imags, o_dim, tensor([0, 5], device=dev))
    diag = torch.index_select(imags, o_dim, tensor([1, 4], device=dev))
    vertic = torch.index_select(imags, o_dim, tensor([2, 3], device=dev))
    deg15i, deg165i = torch.unbind(horiz, dim=o_dim)
    deg45i, deg135i = torch.unbind(diag, dim=o_dim)
    deg75i, deg105i = torch.unbind(vertic, dim=o_dim)

    lh = c2q((deg15r, deg15i,deg165r, deg165i))
    hl = c2q((deg75r, deg75i,deg105r, deg105i))
    hh = c2q((deg45r, deg45i,deg135r, deg135i))

    return lh, hl, hh


def fwd_j1(x, h0, h1, skip_hps, o_dim, mode):
    """ Level 1 forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    # Level 1 forward (biorthogonal analysis filters)
    if not skip_hps:
        lo = first_filter_1d(x, h0, mode, -1)
        hi = first_filter_1d(x, h1, mode, -1)
        ll = first_filter_1d(lo, h0, mode, -2)
        lh = first_filter_1d(lo, h1, mode, -2)
        hl = first_filter_1d(hi, h0, mode, -2)
        hh = first_filter_1d(hi, h1, mode, -2)
        del lo, hi
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = first_filter_1d(x, h0, mode, -1)
        ll = first_filter_1d(ll, h0, mode, -2)
        highr = x.new_zeros([])
        highi = x.new_zeros([])
    return ll, highr, highi


def fwd_j1_3d(x, h0, h1, skip_hps, o_dim, mode):
    """ Level 1 forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    # Level 1 forward (biorthogonal analysis filters)
    if not skip_hps:
        lo = first_filter_1d(x, h0, mode, -1)
        hi = first_filter_1d(x, h1, mode, -1)
        ll = first_filter_1d(lo, h0, mode, -2)
        lh = first_filter_1d(lo, h1, mode, -2)
        hl = first_filter_1d(hi, h0, mode, -2)
        hh = first_filter_1d(hi, h1, mode, -2)
        lll = first_filter_1d(ll, h0, mode, -3)
        llh = first_filter_1d(ll, h1, mode, -3)
        lhl = first_filter_1d(lh, h0, mode, -3)
        lhh = first_filter_1d(lh, h1, mode, -3)
        hll = first_filter_1d(hl, h0, mode, -3)
        hlh = first_filter_1d(hl, h1, mode, -3)
        hhl = first_filter_1d(hh, h0, mode, -3)
        hhh = first_filter_1d(hh, h1, mode, -3)
        del lo, hi, ll, lh, hl, hh
        highr, highi = highs_to_orientations_3d(llh, lhl, lhh, hll, hlh, hhl, hhh, o_dim)
    else:
        lll = first_filter_1d(x, h0, mode, -1)
        lll = first_filter_1d(lll, h0, mode, -2)
        lll = first_filter_1d(lll, h0, mode, -3)
        highr = x.new_zeros([])
        highi = x.new_zeros([])
    return lll, highr, highi

# TODO: Version 1.5
def inv_j1(ll, highr, highi, g0, g1, o_dim, h_dim, w_dim, mode):
    """ Level1 inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)

        if ll is None or ll.shape == torch.Size([]):
            # Interpolate
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode)
            del lh, hh, hl
        else:
            # Possibly cut back some rows to make the ll match the highs
            r, c = ll.shape[2:]
            r1, c1 = highr.shape[h_dim], highr.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]
            # Interpolate
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode) + colfilter(ll, g0, mode)
            del lh, hl, hh

        y = rowfilter(hi, g1, mode) + rowfilter(lo, g0, mode)

    return y

# TODO: Version 1.5
def inv_j1_3d(ll, highr, highi, g0, g1, o_dim, h_dim, w_dim, mode):
    """ Level1 inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)

        if ll is None or ll.shape == torch.Size([]):
            # Interpolate
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode)
            del lh, hh, hl
        else:
            # Possibly cut back some rows to make the ll match the highs
            r, c = ll.shape[2:]
            r1, c1 = highr.shape[h_dim], highr.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:,:,1:-1]
            if c != c1 * 2:
                ll = ll[:,:,:,1:-1]
            # Interpolate
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode) + colfilter(ll, g0, mode)
            del lh, hl, hh

        y = rowfilter(hi, g1, mode) + rowfilter(lo, g0, mode)

    return y


def fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode):
    """ Level 2 plus forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = plus_filter_1d(x, h0b, h0a, False, mode, -1)
        hi = plus_filter_1d(x, h1b, h1a, True, mode, -1)

        ll = plus_filter_1d(lo, h0b, h0a, False, mode, -2)
        lh = plus_filter_1d(lo, h1b, h1a, True, mode, -2)
        hl = plus_filter_1d(hi, h0b, h0a, False, mode, -2)
        hh = plus_filter_1d(hi, h1b, h1a, True, mode, -2)
        del lo, hi
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = plus_filter_1d(x, h0b, h0a, False, mode, -1)
        ll = plus_filter_1d(ll, h0b, h0a, False, mode, -2)
        highr = None
        highi = None

    return ll, highr, highi


def fwd_j2plus_3d(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode):
    """ Level 2 plus forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = plus_filter_1d(x, h0b, h0a, False, mode, -1)
        hi = plus_filter_1d(x, h1b, h1a, True, mode, -1)

        ll = plus_filter_1d(lo, h0b, h0a, False, mode, -2)
        lh = plus_filter_1d(lo, h1b, h1a, True, mode, -2)
        hl = plus_filter_1d(hi, h0b, h0a, False, mode, -2)
        hh = plus_filter_1d(hi, h1b, h1a, True, mode, -2)
        lll = plus_filter_1d(ll, h0b, h0a, False, mode, -3)
        llh = plus_filter_1d(ll, h1b, h1a, True, mode, -3)
        lhl = plus_filter_1d(lh, h0b, h0a, False, mode, -3)
        lhh = plus_filter_1d(lh, h1b, h1a, True, mode, -3)
        hll = plus_filter_1d(hl, h0b, h0a, False, mode, -3)
        hlh = plus_filter_1d(hl, h1b, h1a, True, mode, -3)
        hhl = plus_filter_1d(hh, h0b, h0a, False, mode, -3)
        hhh = plus_filter_1d(hh, h1b, h1a, True, mode, -3)
        del lo, hi, ll, lh, hl, hh
        highr, highi = highs_to_orientations_3d(llh, lhl, lhh, hll, hlh, hhl, hhh, o_dim)
    else:
        lll = plus_filter_1d(x, h0b, h0a, False, mode, -1)
        lll = plus_filter_1d(lll, h0b, h0a, False, mode, -2)
        lll = plus_filter_1d(lll, h0b, h0a, False, mode, -3)
        highr = None
        highi = None

    return lll, highr, highi


#TODO: Version 1.5
def inv_j2plus():
    pass

#TODO: Version 1.5
def inv_j2plus_3d(ll, highr, highi, g0a, g1a, g0b, g1b, o_dim, h_dim, w_dim, mode):
    """ Level2+ inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowifilt(colifilt(ll, g0b, g0a, False, mode), g0b, g0a, False, mode)
    else:
        # Get the double sampled bandpass coefficients
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)

        if ll is None or ll.shape == torch.Size([]):
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True, mode) + \
                colifilt(hl, g0b, g0a, False, mode)
            lo = colifilt(lh, g1b, g1a, True, mode)
            del lh, hh, hl
        else:
            # Interpolate
            hi = colifilt(hh, g1b, g1a, True, mode) + \
                colifilt(hl, g0b, g0a, False, mode)
            lo = colifilt(lh, g1b, g1a, True, mode) + \
                colifilt(ll, g0b, g0a, False, mode)
            del lh, hl, hh

        y = rowifilt(hi, g1b, g1a, True, mode) + \
            rowifilt(lo, g0b, g0a, False, mode)
    return y


class FWD_J1(Function):
    """ Differentiable function doing 1 level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, o_dim, ri_dim, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(h0, h1)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]

        ll, highr, highi = fwd_j1(x, h0, h1, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = ll.new_zeros([])
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0, h1 = ctx.saved_tensors
        mode = ctx.mode
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j1(dl, dhr, dhi, h0, h1, o_dim, h_dim, w_dim, mode)

        return dx, None, None, None, None, None, None


class FWD_J1_3D(Function):
    """ Differentiable function doing 1 level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, o_dim, ri_dim, mode):
        # 3d_ctx : shape = [N, C, D, H, W]
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(h0, h1)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim = o_dim, ri_dim

        lll, highr, highi = fwd_j1_3d(x, h0, h1, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = lll.new_zeros([])
        return lll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0, h1 = ctx.saved_tensors
        mode = ctx.mode
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j1(dl, dhr, dhi, h0, h1, o_dim, h_dim, w_dim, mode)

        return dx, None, None, None, None, None, None


class FWD_J2PLUS(Function):
    """ Differentiable function doing second level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim, mode):
        mode = 'symmetric'
        ctx.mode = mode
        ctx.save_for_backward(h0a, h1a, h0b, h1b)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]

        ll, highr, highi = fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = ll.new_zeros([])
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0a, h1a, h0b, h1b = ctx.saved_tensors
        mode = ctx.mode
        # The colifilt and rowifilt functions use conv2d not conv2d_transpose,
        # so need to reverse the filters
        h0a, h0b = h0b, h0a
        h1a, h1b = h1b, h1a
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j2plus(dl, dhr, dhi, h0a, h1a, h0b, h1b,
                            o_dim, h_dim, w_dim, mode)

        return dx, None, None, None, None, None, None, None, None


class FWD_J2PLUS_3D(Function):
    """ Differentiable function doing second level forward DTCWT """
    @staticmethod
    def forward(ctx, x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim, mode):
        mode = 'symmetric'
        ctx.mode = mode
        ctx.save_for_backward(h0a, h1a, h0b, h1b)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        # o_dim, ri_dim = o_dim, ri_dim

        ll, highr, highi = fwd_j2plus_3d(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = ll.new_zeros([])
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0a, h1a, h0b, h1b = ctx.saved_tensors
        mode = ctx.mode
        # The colifilt and rowifilt functions use conv2d not conv2d_transpose,
        # so need to reverse the filters
        h0a, h0b = h0b, h0a
        h1a, h1b = h1b, h1a
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j2plus(dl, dhr, dhi, h0a, h1a, h0b, h1b,
                            o_dim, h_dim, w_dim, mode)

        return dx, None, None, None, None, None, None, None, None

#TODO: Version 1.5
class INV_J1(Function):
    """ Differentiable function doing 1 level inverse DTCWT """
    @staticmethod
    def forward(ctx, lows, highs, g0, g1, o_dim, ri_dim, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0, g1)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        if highs is not None and highs.shape != torch.Size([]):
            highr, highi = torch.unbind(highs, dim=ri_dim)
        else:
            highr = lows.new_zeros([])
            highi = lows.new_zeros([])
        y = inv_j1(lows, highr, highi, g0, g1, o_dim, h_dim, w_dim, mode)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0, g1 = ctx.saved_tensors
        dl = None
        dh = None
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        mode = ctx.mode
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _, _ = fwd_j1(dy, g0, g1, True, o_dim, mode)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dhr, dhi = fwd_j1(dy, g0, g1, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dhr, dhi = fwd_j1(dy, g0, g1, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)

        return dl, dh, None, None, None, None, None

#TODO: Version 1.5
class INV_J2PLUS(Function):
    """ Differentiable function doing level 2 onwards inverse DTCWT """
    @staticmethod
    def forward(ctx, lows, highs, g0a, g1a, g0b, g1b, o_dim, ri_dim, mode):
        mode = 'symmetric'
        ctx.mode = mode
        ctx.save_for_backward(g0a, g1a, g0b, g1b)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        if highs is not None and highs.shape != torch.Size([]):
            highr, highi = torch.unbind(highs, dim=ri_dim)
        else:
            highr = lows.new_zeros([])
            highi = lows.new_zeros([])
        y = inv_j2plus(lows, highr, highi, g0a, g1a, g0b, g1b,
                       o_dim, h_dim, w_dim, mode)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0a, g1a, g0b, g1b = ctx.saved_tensors
        g0a, g0b = g0b, g0a
        g1a, g1b = g1b, g1a
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        mode = ctx.mode
        dl = None
        dh = None
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _,  _ = fwd_j2plus(dy, g0a, g1a, g0b, g1b, True, o_dim, mode)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dhr, dhi = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dhr, dhi = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)

        return dl, dh, None, None, None, None, None, None, None