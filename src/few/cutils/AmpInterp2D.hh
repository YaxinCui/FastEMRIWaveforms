#ifndef __AMP_INTERP_2D_HH__
#define __AMP_INTERP_2D_HH__

#include "global.h"

void interp2D_wrap(double* z, const double* tx, int nx, const double* ty, int ny, double* c,
             int kx, int ky, const double* x, int mx,
             const double* y, int my, int num_indiv_c, int len_indiv_c);

// 2026-09-04 14:43 CST (linux): Opt-in mixed32 ABI.  Coefficients, knots,
// coordinates, and spline accumulation are FP32, while outputs are promoted to
// FP64 for the existing complex128 spin interpolation and waveform pipeline.
void interp2D_mixed32_wrap(double* z, const float* tx, int nx,
             const float* ty, int ny, float* c, int kx, int ky,
             const float* x, int mx, const float* y, int my,
             int num_indiv_c, int len_indiv_c);

#endif // __AMP_INTERP_2D_HH__
