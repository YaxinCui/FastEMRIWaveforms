#ifndef __INTERP_H__
#define __INTERP_H__

#include "global.h"

#ifdef __CUDACC__
#include "cusparse.h"
#include <stdexcept>

/*
CUDA / cuSPARSE error checking.

These throw std::runtime_error (propagated to Python as RuntimeError via the
`except +` declarations in pyinterp.pyx) instead of exit(-1): a hard exit
kills the whole sampler process with no status and no traceback, and cannot
be intercepted by any Python-level guard. The message carries the actual
status so INVALID_VALUE / ALLOC_FAILED / EXECUTION_FAILED are distinguishable.
*/
#define CUDA_CALL(X)                                                          \
    do                                                                        \
    {                                                                         \
        cudaError_t _few_status = (X);                                        \
        if (_few_status != cudaSuccess)                                       \
        {                                                                     \
            char _few_msg[512];                                               \
            snprintf(_few_msg, sizeof(_few_msg),                              \
                     "CUDA error in %s at %s:%d: %s (status %d)",             \
                     __func__, __FILE__, __LINE__,                            \
                     cudaGetErrorString(_few_status), (int)_few_status);      \
            fprintf(stderr, "%s\n", _few_msg);                                \
            throw std::runtime_error(_few_msg);                               \
        }                                                                     \
    } while (0)

#define CUSPARSE_CALL(X)                                                      \
    do                                                                        \
    {                                                                         \
        cusparseStatus_t _few_status = (X);                                   \
        if (_few_status != CUSPARSE_STATUS_SUCCESS)                           \
        {                                                                     \
            char _few_msg[512];                                               \
            snprintf(_few_msg, sizeof(_few_msg),                              \
                     "cuSPARSE error in %s at %s:%d: %s (status %d)",         \
                     __func__, __FILE__, __LINE__,                            \
                     cusparseGetErrorString(_few_status), (int)_few_status);  \
            fprintf(stderr, "%s\n", _few_msg);                                \
            throw std::runtime_error(_few_msg);                               \
        }                                                                     \
    } while (0)

#endif

void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag);

void get_waveform(cmplx *d_waveform, double *interp_array, double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev);

void get_waveform_generic_fd(cmplx *waveform,
             double *interp_array, double *phase_spline_t, double *phase_spline_coeffs,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *mode_start_inds, int *mode_end_inds, int num_segments,
              cmplx *Ylm_all, int zero_index, bool include_minus_m, bool separate_modes);


#endif // __INTERP_H__
