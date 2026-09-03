"""Cython owner for FEW's explicitly selected Apple Metal mode sum.

2026-09-02 13:50 CST (mac): Keep one native Metal context per backend instance,
translate FEW's existing NumPy pointer ABI, and release the context
deterministically when the backend is discarded.
"""

from libc.stddef cimport size_t

from few.cutils.wrappers import wrapper


cdef extern from "metal_sum.hh":
    const char *few_metal_sum_last_error()
    void *few_metal_sum_context_create()
    void few_metal_sum_context_destroy(void *opaque_context)
    int few_metal_sum_evaluate(
        void *opaque_context,
        double *waveform,
        const double *interpolation,
        const double *phase_times,
        const double *phase_coefficients,
        const int *m_values,
        const int *k_values,
        const int *n_values,
        int init_length,
        int output_length,
        int mode_count,
        const double *ylms,
        double delta_t,
        const double *trajectory_times,
        double *gpu_seconds,
    )


cdef class MetalSummation:
    cdef void *_context
    cdef double _last_gpu_seconds

    def __cinit__(self):
        self._context = few_metal_sum_context_create()
        self._last_gpu_seconds = 0.0
        if self._context == NULL:
            self._raise_last_error("Metal context creation failed")

    cdef void _raise_last_error(self, str fallback) except *:
        cdef const char *message = few_metal_sum_last_error()
        if message == NULL:
            raise RuntimeError(fallback)
        raise RuntimeError((<bytes>message).decode("utf-8"))

    def close(self):
        """Release the native Metal queue and runtime-compiled pipeline."""
        if self._context != NULL:
            few_metal_sum_context_destroy(self._context)
            self._context = NULL

    def __dealloc__(self):
        if self._context != NULL:
            few_metal_sum_context_destroy(self._context)
            self._context = NULL

    @property
    def closed(self):
        return self._context == NULL

    @property
    def last_gpu_seconds(self):
        return self._last_gpu_seconds

    def get_waveform_wrap(
        self,
        waveform,
        interpolation,
        phase_times,
        phase_coefficients,
        m_values,
        k_values,
        n_values,
        init_length,
        output_length,
        mode_count,
        ylms,
        delta_t,
        trajectory_times,
        _device,
    ):
        """Evaluate FEW's existing 14-argument time-domain summation ABI."""
        cdef list target_args
        cdef dict target_kwargs
        cdef size_t waveform_pointer
        cdef size_t interpolation_pointer
        cdef size_t phase_times_pointer
        cdef size_t phase_coefficients_pointer
        cdef size_t m_values_pointer
        cdef size_t k_values_pointer
        cdef size_t n_values_pointer
        cdef size_t ylms_pointer
        cdef size_t trajectory_times_pointer
        cdef int status

        if self._context == NULL:
            raise RuntimeError("Metal summation context is closed")

        target_args, target_kwargs = wrapper(
            waveform,
            interpolation,
            phase_times,
            phase_coefficients,
            m_values,
            k_values,
            n_values,
            init_length,
            output_length,
            mode_count,
            ylms,
            delta_t,
            trajectory_times,
            _device,
        )
        waveform_pointer = target_args[0]
        interpolation_pointer = target_args[1]
        phase_times_pointer = target_args[2]
        phase_coefficients_pointer = target_args[3]
        m_values_pointer = target_args[4]
        k_values_pointer = target_args[5]
        n_values_pointer = target_args[6]
        ylms_pointer = target_args[10]
        trajectory_times_pointer = target_args[12]
        status = few_metal_sum_evaluate(
            self._context,
            <double *>waveform_pointer,
            <const double *>interpolation_pointer,
            <const double *>phase_times_pointer,
            <const double *>phase_coefficients_pointer,
            <const int *>m_values_pointer,
            <const int *>k_values_pointer,
            <const int *>n_values_pointer,
            int(target_args[7]),
            int(target_args[8]),
            int(target_args[9]),
            <const double *>ylms_pointer,
            float(target_args[11]),
            <const double *>trajectory_times_pointer,
            &self._last_gpu_seconds,
        )
        if status != 0:
            self._raise_last_error("Metal summation failed")
