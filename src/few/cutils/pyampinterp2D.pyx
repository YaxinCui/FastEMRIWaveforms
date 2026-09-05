import numpy as np
cimport numpy as np
from libcpp.string cimport string

from few.cutils.wrappers import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "AmpInterp2D.hh":
    void interp2D_wrap(double *z, const double* tx, int nx, const double* ty, int ny, double* c,
             int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c) except+;
    # 2026-09-04 14:44 CST (linux): Expose the explicit FP32-input/
    # FP64-output spline candidate without changing the accepted FP64 symbol.
    void interp2D_mixed32_wrap(double *z, const float* tx, int nx,
             const float* ty, int ny, float* c, int kx, int ky,
             const float* x, int mx, const float* y, int my,
             int num_indiv_c, int len_indiv_c) except+;


@pointer_adjust
def interp2D(z, tx, nx, ty, ny, c, kx, ky, x, mx, y, my, num_indiv_c, len_indiv_c):

    cdef size_t z_in = z
    cdef size_t c_in = c
    cdef size_t tx_in = tx
    cdef size_t ty_in = ty
    cdef size_t x_in = x
    cdef size_t y_in = y

    interp2D_wrap(<double*> z_in, <double*> tx_in, nx, <double*> ty_in, ny, <double*> c_in,
            kx, ky, <double*> x_in, mx,
            <double*> y_in, my, num_indiv_c, len_indiv_c);


@pointer_adjust
def interp2D_mixed32(z, tx, nx, ty, ny, c, kx, ky, x, mx, y, my, num_indiv_c, len_indiv_c):

    cdef size_t z_in = z
    cdef size_t c_in = c
    cdef size_t tx_in = tx
    cdef size_t ty_in = ty
    cdef size_t x_in = x
    cdef size_t y_in = y

    interp2D_mixed32_wrap(<double*> z_in, <float*> tx_in, nx,
            <float*> ty_in, ny, <float*> c_in, kx, ky,
            <float*> x_in, mx, <float*> y_in, my,
            num_indiv_c, len_indiv_c);
