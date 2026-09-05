import numpy as np
cimport numpy as np
from libcpp cimport bool

from few.cutils.wrappers import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "interpolate.hh":
    ctypedef void* cmplx 'cmplx'
    void interpolate_arrays(double *t_arr, double *interp_array, int ninterps, int length, double *B, double *upper_diag, double *diag, double *lower_diag)

    void get_waveform(cmplx *d_waveform, double *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 14:54 CST (linux): Bind the isolated same-FP64 scheduling
    # control and range-reduced mixed32 phase-exponential candidate.
    void get_waveform_optimized(cmplx *d_waveform, double *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev)

    void get_waveform_mixed32(cmplx *d_waveform, double *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, cmplx *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 15:09 CST (linux): Forward float amplitude splines and opaque
    # complex64 Ylms to the full mixed32 mode-evaluation kernel.
    void get_waveform_mixed32_full(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 15:31 CST (linux): Bind the integer-phasor mixed32 candidate.
    void get_waveform_mixed32_recurrence(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 15:39 CST (linux): Bind block-local FP32 accumulation variant.
    void get_waveform_mixed32_fast(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 17:01 CST (linux): Bind fast-intrinsic controls with either
    # FP64 or block-local FP32 mode accumulation.
    void get_waveform_mixed32_intrinsic(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    void get_waveform_mixed32_intrinsic_fast(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    # 2026-09-04 16:34 CST (linux): Bind the warp-per-output CUDA candidate.
    void get_waveform_mixed32_warp(cmplx *d_waveform, float *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
                  int *d_m, int *d_k, int *d_n, int init_len, int out_len, int num_teuk_modes, void *d_Ylms,
                  double delta_t, double *h_t, int dev)

    void get_waveform_generic_fd(cmplx *waveform,
             double *interp_array, double *phase_interp_t, double *phase_interp_coeffs,
              int *m_arr_in, int *k_arr_in, int *n_arr_in, int num_teuk_modes,
              double delta_t, double *old_time_arr, int init_length, int data_length,
              double *frequencies, int *mode_start_inds, int *mode_end_inds, int num_segments, cmplx *Ylm_all, int zero_index,
              bool include_minus_m, bool separate_modes);


def interpolate_arrays_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    t_arr, interp_array, ninterps, length, B, upper_diag, diag, lower_diag = targs

    cdef size_t t_arr_in = t_arr
    cdef size_t interp_array_in = interp_array
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate_arrays(<double *>t_arr_in, <double *>interp_array_in, ninterps, length, <double *>B_in, <double *>upper_diag_in, <double *>diag_in, <double *>lower_diag_in)


def get_waveform_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform(<cmplx *>d_waveform_in, <double *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <cmplx *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


def get_waveform_optimized_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_optimized(<cmplx *>d_waveform_in, <double *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <cmplx *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


def get_waveform_mixed32_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_mixed32(<cmplx *>d_waveform_in, <double *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <cmplx *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


def get_waveform_mixed32_full_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_mixed32_full(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


# 2026-09-04 15:31 CST (linux): Keep a distinct Python entry point so tests and
# benchmarks can compare the integer-phasor path with mixed32_full.
def get_waveform_mixed32_recurrence_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_mixed32_recurrence(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


# 2026-09-04 15:39 CST (linux): Expose the FP32 mode-block accumulation only
# through an explicit experimental selector.
def get_waveform_mixed32_fast_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_mixed32_fast(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


# 2026-09-04 17:01 CST (linux): Fast intrinsic with per-mode FP64 accumulation.
def get_waveform_mixed32_intrinsic_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)
    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs
    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t
    get_waveform_mixed32_intrinsic(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


# 2026-09-04 17:01 CST (linux): Fast intrinsic with block-local FP32 reduction.
def get_waveform_mixed32_intrinsic_fast_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)
    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs
    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t
    get_waveform_mixed32_intrinsic_fast(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


# 2026-09-04 16:34 CST (linux): Expose the warp-parallel kernel through an
# explicit selector; the default FP64 wrapper is unchanged.
def get_waveform_mixed32_warp_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (d_waveform, interp_array, phase_interp_t, phase_interp_coeffs,
              d_m, d_k, d_n, init_len, out_len, num_teuk_modes, d_Ylms,
              delta_t, h_t, dev) = targs

    cdef size_t d_waveform_in = d_waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t d_m_in = d_m
    cdef size_t d_k_in = d_k
    cdef size_t d_n_in = d_n
    cdef size_t d_Ylms_in = d_Ylms
    cdef size_t h_t_in = h_t

    get_waveform_mixed32_warp(<cmplx *>d_waveform_in, <float *>interp_array_in, <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
                <int *>d_m_in, <int *>d_k_in, <int *>d_n_in, init_len, out_len, num_teuk_modes, <void *>d_Ylms_in,
                delta_t, <double *>h_t_in, dev)


def get_waveform_generic_fd_wrap(*args, **kwargs):

    targs, kwargs = wrapper(*args, **kwargs)

    (waveform,
             interp_array, phase_interp_t, phase_interp_coeffs,
              m_arr_in, k_arr_in, n_arr_in, num_teuk_modes,
              delta_t, old_time_arr, init_length, data_length,
              frequencies, mode_start_inds, mode_end_inds, num_segments, Ylm_all, zero_index, include_minus_m, separate_modes) = targs

    cdef size_t waveform_in = waveform
    cdef size_t interp_array_in = interp_array
    cdef size_t phase_interp_t_in = phase_interp_t
    cdef size_t phase_interp_coeffs_in = phase_interp_coeffs
    cdef size_t m_arr_in_in = m_arr_in
    cdef size_t k_arr_in_in = k_arr_in
    cdef size_t n_arr_in_in = n_arr_in
    cdef size_t old_time_arr_in = old_time_arr
    cdef size_t frequencies_in = frequencies
    cdef size_t mode_start_inds_in = mode_start_inds
    cdef size_t mode_end_inds_in = mode_end_inds
    cdef size_t Ylm_all_in = Ylm_all

    get_waveform_generic_fd(<cmplx *>waveform_in,
              <double *>interp_array_in,  <double *>phase_interp_t_in, <double *>phase_interp_coeffs_in,
              <int *>m_arr_in_in, <int *>k_arr_in_in, <int *>n_arr_in_in, num_teuk_modes,
              delta_t, <double *>old_time_arr_in, init_length, data_length,
              <double *>frequencies_in, <int *>mode_start_inds_in, <int *>mode_end_inds_in, num_segments,
              <cmplx *>Ylm_all_in, zero_index, include_minus_m, separate_modes)
