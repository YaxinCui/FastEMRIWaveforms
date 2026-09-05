#ifndef __INTERP_H__
#define __INTERP_H__

#include "global.h"

#ifdef __CUDACC__
#include "cusparse.h"

/*
CuSparse error checking
*/
#define ERR_NE(X, Y)                                                                 \
    do                                                                               \
    {                                                                                \
        if ((X) != (Y))                                                              \
        {                                                                            \
            fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
            exit(-1);                                                                \
        }                                                                            \
    } while (0)

#define CUDA_CALL(X) ERR_NE((X), cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

#endif

void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag);

void get_waveform(cmplx *d_waveform, double *interp_array, double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev);

// 2026-09-04 14:52 CST (linux): Opt-in launch/accumulation candidate with the
// accepted FP64 phase exponential, plus a mixed32 phase-exponential candidate.
// Both retain FP64 spline inputs, accumulation, and complex128 output.
void get_waveform_optimized(cmplx *d_waveform, double *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, cmplx *d_Ylms, double delta_t,
              double *h_t, int dev);

void get_waveform_mixed32(cmplx *d_waveform, double *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, cmplx *d_Ylms, double delta_t,
              double *h_t, int dev);

// 2026-09-04 15:04 CST (linux): Full mixed32 mode-evaluation candidate.  Phase
// spline evaluation and waveform accumulation remain FP64; amplitude-spline
// storage/evaluation, reduced phase combination, phasors, and Ylms are FP32.
void get_waveform_mixed32_full(cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

// 2026-09-04 15:31 CST (linux): Mixed32 summation variant that replaces the
// per-mode trigonometric evaluation with integer powers of three base phasors.
void get_waveform_mixed32_recurrence(cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

// 2026-09-04 15:39 CST (linux): Exploratory recurrence variant with block-local
// complex64 mode accumulation and one complex128 promotion per mode block.
void get_waveform_mixed32_fast(cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

// 2026-09-04 17:01 CST (linux): Range-reduced CUDA fast-intrinsic variants,
// with FP64 and block-local FP32 accumulation respectively.
void get_waveform_mixed32_intrinsic(cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

void get_waveform_mixed32_intrinsic_fast(
              cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

// 2026-09-04 16:34 CST (linux): CUDA warp-per-output mixed32 summation path.
void get_waveform_mixed32_warp(cmplx *d_waveform, float *interp_array,
              double *phase_spline_t, double *phase_spline_coeffs,
              int *d_m, int *d_k, int *d_n, int init_len, int out_len,
              int num_teuk_modes, void *d_Ylms, double delta_t,
              double *h_t, int dev);

void get_waveform_generic_fd(cmplx *waveform,
             double *interp_array, double *phase_spline_t, double *phase_spline_coeffs,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *mode_start_inds, int *mode_end_inds, int num_segments,
              cmplx *Ylm_all, int zero_index, bool include_minus_m, bool separate_modes);


#endif // __INTERP_H__
