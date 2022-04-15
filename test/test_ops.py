# Owner(s): ["module: unknown"]

from collections.abc import Sequence
from functools import partial
import warnings
import unittest
import itertools
import torch

from torch.testing import make_tensor
from torch.testing._internal.common_dtype import floating_and_complex_types_and, all_types_and_complex_and
from torch.testing._internal.common_utils import \
    (TestCase, is_iterable_of_tensors, run_tests, IS_SANDCASTLE, clone_input_helper,
     IS_IN_CI, suppress_warnings, noncontiguous_like,
     TEST_WITH_ASAN, IS_WINDOWS, IS_FBCODE, first_sample)
from torch.testing._internal.common_methods_invocations import \
    (op_db, _NOTHING, UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo)
from torch.testing._internal.common_device_type import \
    (deviceCountAtLeast, instantiate_device_type_tests, ops, onlyCPU,
     onlyCUDA, onlyNativeDeviceTypes, OpDTypes, skipMeta)


import torch.testing._internal.opinfo_helper as opinfo_helper
from torch.testing._internal import composite_compliance

TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None

# TODO: fixme https://github.com/pytorch/pytorch/issues/68972
torch.set_default_dtype(torch.float32)

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(ops, dtypes=OpDTypes.supported,
                       allowed_dtypes=(torch.float, torch.cfloat))

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for Unary Ufuncs (separately implemented in test/test_unary_ufuncs.py)
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = list(filter(lambda op: not isinstance(op, (UnaryUfuncInfo, ReductionOpInfo,
                     SpectralFuncInfo)) and op.ref is not None and op.ref is not _NOTHING, op_db))


# Tests that apply to all operators and aren't related to any particular
#   system
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_IN_CI:
            err_msg = ("The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                       "This is OK for testing, but be sure to set the dtypes manually before landing your PR!")
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo_helper.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo_helper.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # Validates that each OpInfo specifies its forward and backward dtypes
    #   correctly for CPU and CUDA devices
    @skipMeta
    @onlyNativeDeviceTypes
    @ops(op_db, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        # Check complex32 support only if the op claims.
        # TODO: Once the complex32 support is better, we should add check for complex32 unconditionally.
        include_complex32 = ((torch.complex32,) if op.supports_dtype(torch.complex32, device) else ())

        # dtypes to try to backward in
        allowed_backward_dtypes = floating_and_complex_types_and(
            *((torch.half, torch.bfloat16) + include_complex32))

        # lists for (un)supported dtypes
        supported_dtypes = []
        unsupported_dtypes = []
        supported_backward_dtypes = []
        unsupported_backward_dtypes = []

        def unsupported(dtype):
            unsupported_dtypes.append(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.append(dtype)

        for dtype in all_types_and_complex_and(
                *((torch.half, torch.bfloat16, torch.bool) + include_complex32)):
            # tries to acquire samples - failure indicates lack of support
            requires_grad = (dtype in allowed_backward_dtypes and op.supports_autograd)
            try:
                samples = list(op.sample_inputs(device, dtype, requires_grad=requires_grad))
            except Exception as e:
                unsupported(dtype)
                continue

            # Counts number of successful backward attempts
            # NOTE: This exists as a kludge because this only understands how to
            #   request a gradient if the output is a tensor or a sequence with
            #   a tensor as its first element.
            num_backward_successes = 0
            for sample in samples:
                # tries to call operator with the sample - failure indicates
                #   lack of support
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                except Exception as e:
                    # NOTE: some ops will fail in forward if their inputs
                    #   require grad but they don't support computing the gradient
                    #   in that type! This is a bug in the op!
                    unsupported(dtype)

                # Short-circuits testing this dtype -- it doesn't work
                if dtype in unsupported_dtypes:
                    break

                # Short-circuits if the dtype isn't a backward dtype or
                #   it's already identified as not supported
                if dtype not in allowed_backward_dtypes or dtype in unsupported_backward_dtypes:
                    continue

                # Checks for backward support in the same dtype
                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(result[0], torch.Tensor):
                        backward_tensor = result[0]
                    else:
                        continue

                    # Note: this grad may not have the same dtype as dtype
                    # For functions like complex (float -> complex) or abs
                    #   (complex -> float) the grad tensor will have a
                    #   different dtype than the input.
                    #   For simplicity, this is still modeled as these ops
                    #   supporting grad in the input dtype.
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    num_backward_successes += 1
                except Exception as e:
                    unsupported_backward_dtypes.append(dtype)

            if dtype not in unsupported_dtypes:
                supported_dtypes.append(dtype)
            if num_backward_successes > 0 and dtype not in unsupported_backward_dtypes:
                supported_backward_dtypes.append(dtype)

        # Checks that dtypes are listed correctly and generates an informative
        #   error message
        device_type = torch.device(device).type
        claimed_supported = set(op.supported_dtypes(device_type))
        supported_dtypes = set(supported_dtypes)
        supported_but_unclaimed = supported_dtypes - claimed_supported
        claimed_but_unsupported = claimed_supported - supported_dtypes
        msg = """The supported dtypes for {0} on {1} according to its OpInfo are
        {2}, but the detected supported dtypes are {3}.
        """.format(op.name, device_type, claimed_supported, supported_dtypes)

        if len(supported_but_unclaimed) > 0:
            msg += "The following dtypes should be added to the OpInfo: {0}. ".format(supported_but_unclaimed)
        if len(claimed_but_unsupported) > 0:
            msg += "The following dtypes should be removed from the OpInfo: {0}.".format(claimed_but_unsupported)

        self.assertEqual(supported_dtypes, claimed_supported, msg=msg)

        # Checks that backward dtypes are listed correctly and generates an
        #   informative error message
        # NOTE: this code is nearly identical to the check + msg generation
        claimed_backward_supported = set(op.supported_backward_dtypes(device_type))
        supported_backward_dtypes = set(supported_backward_dtypes)

        supported_but_unclaimed = supported_backward_dtypes - claimed_backward_supported
        claimed_but_unsupported = claimed_backward_supported - supported_backward_dtypes
        msg = """The supported backward dtypes for {0} on {1} according to its OpInfo are
        {2}, but the detected supported backward dtypes are {3}.
        """.format(op.name, device_type, claimed_backward_supported, supported_backward_dtypes)

        if len(supported_but_unclaimed) > 0:
            msg += "The following backward dtypes should be added to the OpInfo: {0}. ".format(supported_but_unclaimed)
        if len(claimed_but_unsupported) > 0:
            msg += "The following backward dtypes should be removed from the OpInfo: {0}.".format(claimed_but_unsupported)

        self.assertEqual(supported_backward_dtypes, claimed_backward_supported, msg=msg)

    # Validates that each OpInfo works correctly on different CUDA devices
    @onlyCUDA
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for cuda_device_str in devices:
            cuda_device = torch.device(cuda_device_str)
            # NOTE: only tests on first sample
            samples = op.sample_inputs(cuda_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == cuda_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(map(lambda t: t.device == cuda_device, result)))
            else:
                self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

    # Tests that the function and its (ndarray-accepting) reference produce the same
    #   values on the tensors from sample_inputs func for the corresponding op.
    # This test runs in double and complex double precision because
    # NumPy does computation internally using double precision for many functions
    # resulting in possible equality check failures.
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_reference_testing(self, device, dtype, op):
        try:
            # Sets the default dtype to NumPy's default dtype of double
            cur_default = torch.get_default_dtype()
            torch.set_default_dtype(torch.double)
            reference_inputs = op.reference_inputs(device, dtype)
            for sample_input in reference_inputs:
                self.compare_with_reference(op, op.ref, sample_input, exact_dtype=(dtype is not torch.long))
        finally:
            torch.set_default_dtype(cur_default)

    @skipMeta
    @onlyNativeDeviceTypes
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                op(si.input, *si.args, **si.kwargs)

    # Tests that the function produces the same result when called with
    #   noncontiguous tensors.
    # TODO: get working with Windows by addressing failing operators
    # TODO: get working with ASAN by addressing failing operators
    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    def test_noncontiguous_samples(self, device, dtype, op):
        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            n_inp, n_args, n_kwargs = sample_input.noncontiguous()

            # Verifies sample input tensors should have no grad or history
            sample_tensor = t_inp if isinstance(t_inp, torch.Tensor) else t_inp[0]
            assert sample_tensor.grad is None
            assert sample_tensor.grad_fn is None

            # validates forward
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)

            self.assertEqual(actual, expected)

            # Validate backward
            # Short-circuits if the op doesn't support grad in this device x dtype
            if not test_grad:
                continue

            expected = sample_input.output_process_fn_grad(expected)
            actual = sample_input.output_process_fn_grad(actual)

            if isinstance(expected, torch.Tensor):
                grad_for_expected = torch.randn_like(expected)
                grad_for_actual = noncontiguous_like(grad_for_expected)
            elif isinstance(expected, Sequence):
                # Filter output elements that do not require grad
                expected = [t for t in expected
                            if isinstance(t, torch.Tensor) and t.requires_grad]
                actual = [n for n in actual
                          if isinstance(n, torch.Tensor) and n.requires_grad]
                grad_for_expected = [torch.randn_like(t) for t in expected]
                grad_for_actual = [noncontiguous_like(n) for n in grad_for_expected]
            else:
                # Nothing to do if it returns a scalar or things like that
                continue

            # Concatenate inputs into a tuple
            t_inputs = (t_inp,) + t_args if isinstance(t_inp, torch.Tensor) else tuple(t_inp) + t_args
            n_inputs = (n_inp,) + n_args if isinstance(n_inp, torch.Tensor) else tuple(n_inp) + n_args

            # Filter the elemnts that are tensors that require grad
            t_input_tensors = [t for t in t_inputs if isinstance(t, torch.Tensor) and t.requires_grad]
            n_input_tensors = [n for n in n_inputs if isinstance(n, torch.Tensor) and n.requires_grad]

            self.assertEqual(len(t_input_tensors), len(n_input_tensors))

            # Some functions may not use all the inputs to generate gradients. One of the
            # few examples of this "odd" behaviour is F.hinge_embedding_loss
            t_grads = torch.autograd.grad(expected, t_input_tensors, grad_for_expected, allow_unused=True)
            n_grads = torch.autograd.grad(actual, n_input_tensors, grad_for_actual, allow_unused=True)

            msg = "Got different gradients for contiguous / non-contiguous inputs wrt input {}."
            for i, (t, n) in enumerate(zip(t_grads, n_grads)):
                self.assertEqual(t, n, msg=msg.format(i))

    # Separates one case from the following test_out because many ops don't properly implement the
    #   incorrectly sized out parameter warning properly yet
    # Cases test here:
    #   - out= with the correct dtype and device, but the wrong shape
    @ops(op_db, dtypes=OpDTypes.none)
    def test_out_warning(self, device, op):
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]

        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(expected, include_empty=True):
                self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(lambda t: t.stride(), out))

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != 'cpu' and self.device_type != 'cuda':
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(lambda t: t.data_ptr(), out))

            @suppress_warnings
            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)

                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = "Strides are not the same! Original strides were {0} and strides are now {1}".format(
                        original_strides, final_strides)
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case Zero: out= with the correct dtype and device, but the wrong shape
            #   Expected behavior: if nonempty, resize with a warning.
            def _case_zero_transform(t):
                wrong_shape = list(t.shape)

                if len(wrong_shape) == 0:
                    # Handles scalar tensor case (empty list)
                    wrong_shape = [2]
                else:
                    wrong_shape[-1] = wrong_shape[-1] + 1
                return make_tensor(wrong_shape, dtype=t.dtype, device=t.device)

            # Verifies the out values are correct
            _compare_out(_case_zero_transform, compare_strides_and_data_ptrs=False)

            # Additionally validates that the appropriate warning is thrown if a nonempty
            #   tensor is resized.
            def _any_nonempty(out):
                if isinstance(out, torch.Tensor):
                    return out.numel() > 0

                return any(x.numel() > 0 for x in out)

            out = _apply_out_transform(_case_zero_transform, expected)
            msg_fail = "Resized a non-empty tensor but did not warn about it."
            if _any_nonempty(out):
                with self.assertWarnsRegex(UserWarning, "An output with one or more elements", msg=msg_fail):
                    op_out(out=out)

    # Validates ops implement the correct out= behavior
    # See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    #   for a description of the correct behavior
    # Validates the following cases:
    #   - Case 0: out has the correct shape, dtype, and device but is full of extremal values
    #   - Case 1: out has the correct shape, dtype, and device but is noncontiguous
    #   - Case 2: out has the correct dtype and device, but is zero elements
    #   - Case 3: out has the correct shape and dtype, but is on a different device type
    #   - Case 4: out has the with correct shape and device, but a dtype that cannot
    #       "safely" cast to
    @ops(op_db, dtypes=OpDTypes.none)
    def test_out(self, device, op):
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]

        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(expected, include_empty=True):
                self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(lambda t: t.stride(), out))

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != 'cpu' and self.device_type != 'cuda':
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(lambda t: t.data_ptr(), out))

            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)

                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = "Strides are not the same! Original strides were {0} and strides are now {1}".format(
                        original_strides, final_strides)
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case 0: out= with the correct shape, dtype, and device
            #   but NaN values for floating point and complex tensors, and
            #   maximum values for integer tensors.
            #   Expected behavior: out= values have no effect on the computation.
            def _case_zero_transform(t):
                try:
                    info = torch.iinfo(t.dtype)
                    return torch.full_like(t, info.max)
                except TypeError as te:
                    # for non-integer types fills with NaN
                    return torch.full_like(t, float('nan'))

            _compare_out(_case_zero_transform)

            # Case 1: out= with the correct shape, dtype, and device,
            #   but noncontiguous.
            #   Expected behavior: strides are respected and `out` storage is not changed.
            def _case_one_transform(t):
                return make_tensor(t.shape,
                                   dtype=t.dtype,
                                   device=t.device,
                                   noncontiguous=True)

            _compare_out(_case_one_transform)

            # Case 2: out= with the correct dtype and device, but has no elements.
            #   Expected behavior: resize without warning.
            def _case_two_transform(t):
                return make_tensor((0,), dtype=t.dtype, device=t.device)

            _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

            # Also validates that no warning is thrown when this out is resized
            out = _apply_out_transform(_case_two_transform, expected)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                op_out(out=out)

            # Verifies no warning is a resize warning
            for w in caught:
                if "An output with one or more elements" in str(w.message):
                    self.fail("Resizing an out= argument with no elements threw a resize warning!")

            # Case 3: out= with correct shape and dtype, but wrong device.
            wrong_device = None
            if torch.device(device).type != 'cpu':
                wrong_device = 'cpu'
            elif torch.cuda.is_available():
                wrong_device = 'cuda'

            if wrong_device is not None:
                def _case_three_transform(t):
                    return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)

                out = _apply_out_transform(_case_three_transform, expected)
                msg_fail = f"Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}"
                with self.assertRaises(RuntimeError, msg=msg_fail):
                    op_out(out=out)

            # Case 4: out= with correct shape and device, but a dtype
            #   that output cannot be "safely" cast to (long).
            #   Expected behavior: error.
            # NOTE: this case is filtered by dtype since some ops produce
            #   bool tensors, for example, which can be safely cast to any
            #   dtype. It is applied when single tensors are floating point or complex
            #   dtypes, or if an op returns multiple tensors when at least one such
            #   tensor is a floating point or complex dtype.
            _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
            if (isinstance(expected, torch.Tensor) and expected.dtype in _dtypes or
                    (not isinstance(expected, torch.Tensor) and any(t.dtype in _dtypes for t in expected))):
                def _case_four_transform(t):
                    return make_tensor(t.shape, dtype=torch.long, device=t.device)

                out = _apply_out_transform(_case_four_transform, expected)
                msg_fail = "Expected RuntimeError when doing an unsafe cast!"
                msg_fail = msg_fail if not isinstance(expected, torch.Tensor) else \
                    ("Expected RuntimeError when doing an unsafe cast from a result of dtype "
                        f"{expected.dtype} into an out= with dtype torch.long")
                with self.assertRaises(RuntimeError, msg=msg_fail):
                    op_out(out=out)

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (method, inplace)
    #   against eager's gold standard op function variant
    @_variant_ops(op_db)
    def test_variant_consistency_eager(self, device, dtype, op):
        # Acquires variants (method variant, inplace variant, aliases)

        method = op.method_variant
        inplace = op.inplace_variant

        # list of all inplace ops: inplace variant + alias inplace variants if exist
        inplace_ops = [inplace, ]
        variants = [method, inplace]

        for a_op in op.aliases:
            variants.append(a_op.op)
            variants.append(a_op.method_variant)
            variants.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)

        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants))

        _requires_grad = (op.supports_autograd and
                          (dtype.is_floating_point or op.supports_complex_autograd(torch.device(device).type)))

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad, include_conjugated_inputs=include_conjugated_inputs)
        samples = list(samples)

        def _test_consistency_helper(samples, variants):
            for sample in samples:
                # TODO: Check grad for all Tensors requiring grad if sample.input is TensorList
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]

                # Computes function forward and backward values
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None

                output_process_fn_grad = sample.output_process_fn_grad if sample.output_process_fn_grad \
                    else lambda x: x

                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                skip_inplace = False
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is not tensor.dtype):
                    skip_inplace = True

                # TODO: backward consistency only supported for single tensor outputs
                # TODO: backward consistency only checked on sample.input, not all
                #   tensor inputs
                # TODO: update to handle checking grads of all tensor inputs as
                #   derived from each tensor output
                if (op.supports_autograd and isinstance(expected_forward, torch.Tensor)
                        and (dtype.is_floating_point or op.supports_complex_autograd(torch.device(device).type))):
                    output_process_fn_grad(expected_forward).sum().backward()
                    expected_grad = tensor.grad

                # Test eager consistency
                for variant in variants:
                    # Skips inplace ops
                    if variant in inplace_ops and skip_inplace:
                        continue

                    # Compares variant's forward
                    # Note: copies the to-be-modified input when testing the inplace variant
                    tensor.grad = None
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input

                    if variant in inplace_ops and sample.broadcasts_input:
                        with self.assertRaises(RuntimeError,
                                               msg=('inplace variant either incorrectly allowed '
                                                    'resizing or you have marked the sample {}'
                                                    ' incorrectly with `broadcasts_self=True'.format(sample.summary()))):
                            variant_forward = variant(cloned,
                                                      *sample.args,
                                                      **sample.kwargs)
                        continue

                    variant_forward = variant(cloned,
                                              *sample.args,
                                              **sample.kwargs)
                    self.assertEqual(expected_forward, variant_forward)

                    # Compares variant's backward
                    if expected_grad is not None and \
                            (variant not in inplace_ops or op.supports_inplace_autograd):
                        output_process_fn_grad(variant_forward).sum().backward()
                        self.assertEqual(expected_grad, tensor.grad)

        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:
                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                skip_inplace = False
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is not tensor.dtype):
                    skip_inplace = True
                if skip_inplace:
                    return
                for variant in variants:
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input
                    inp_tensor = cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    data_ptr = inp_tensor.data_ptr()
                    variant_forward = variant(cloned,
                                              *sample.args,
                                              **sample.kwargs)
                    # TODO Support non-tensor outputs if they exist for inplace ops
                    if (isinstance(variant_forward, torch.Tensor)):
                        self.assertEqual(data_ptr, variant_forward.data_ptr(), atol=0, rtol=0)
                    else:
                        self.assertTrue(False, "Non-tensor outputs for inplace ops are not supported")

        if len(inplace_ops) > 0:
            inplace_samples = list(filter(lambda sample: not sample.broadcasts_input, samples))
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)

    @onlyCPU
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_floating_inputs_are_differentiable(self, device, dtype, op):
        # Nothing to check if the operation it's not differentiable
        if not op.supports_autograd:
            return

        floating_dtypes = list(floating_and_complex_types_and(torch.bfloat16, torch.float16))

        def check_tensor_floating_is_differentiable(t):
            if isinstance(t, torch.Tensor) and t.dtype in floating_dtypes:
                msg = (f"Found a sampled tensor of floating-point dtype {t.dtype} sampled with "
                       "requires_grad=False. If this is intended, please skip/xfail this test. "
                       "Remember that sampling operations are executed under a torch.no_grad contextmanager.")
                self.assertTrue(t.requires_grad, msg)

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            check_tensor_floating_is_differentiable(sample.input)
            for arg in sample.args:
                check_tensor_floating_is_differentiable(arg)
            for arg in sample.kwargs.values():
                check_tensor_floating_is_differentiable(arg)

    # Reference testing for operations in complex32 against complex64.
    # NOTE: We test against complex64 as NumPy doesn't have a complex32 equivalent dtype.
    @ops(op_db, allowed_dtypes=(torch.complex32,))
    def test_complex_half_reference_testing(self, device, dtype, op):
        if not op.supports_dtype(torch.complex32, device):
            unittest.skip("Does not support complex32")

        for sample in op.sample_inputs(device, dtype):
            actual = op(sample.input, *sample.args, **sample.kwargs)
            (inp, args, kwargs) = sample.transform(lambda x: x.to(torch.complex64))
            expected = op(inp, *args, **kwargs)
            self.assertEqual(actual, expected, exact_dtype=False)

class TestCompositeCompliance(TestCase):
    # Checks if the operator (if it is composite) is written to support most
    # backends and Tensor subclasses. See "CompositeImplicitAutograd Compliance"
    # in aten/src/ATen/native/README.md for more details
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_with_mode(op, args, kwargs)
            composite_compliance.check_all_permutations(op, args, kwargs)

    # There are some weird unexpected successe here that imply rocm goes down
    # a different path than CUDA sometimes. There's not an easy way to describe
    # this in OpInfo so we're just going to skip all ROCM tests...
    @unittest.skipIf(TEST_ROCM, "The CUDA tests give sufficient signal")
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_backward_formula(op, args, kwargs)


class TestMathBits(TestCase):
    # Tests that
    # 1. The operator's output for physically conjugated/negated tensors and conjugate/negative view tensors
    # produces the same value
    # 2. The gradients are same in both cases mentioned in (1)
    # 3. If the operator's inplace variant is supported, tests that the inplace operation
    #    produces the correct value when called on a conjugate/negative view tensor and that the output
    #    has its conj/neg bit set to true
    # This test only runs for C -> R and C -> C functions
    # TODO: add tests for `R->C` functions
    # Note: This test runs for functions that take both tensors and tensorlists as input.
    def _test_math_view(self, device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, out_type):
        inplace_variant = op.inplace_variant

        # helper function to clone and conjugate/negate the input if its a tensor
        # else clone the sequence and conjugate/negate the first element in the sequence
        # If a requires_grad argument is provided the tensor being conjugated/negated will
        # have its requires_grad set to that value.
        def clone_and_perform_view(input, **kwargs):
            if isinstance(input, torch.Tensor):
                requires_grad = kwargs.get('requires_grad', input.requires_grad)
                with torch.no_grad():
                    # Ensure view represents the original sample input
                    input = math_op_physical(input)
                # Note: .conj() is not called under no_grad mode since it's not allowed to modify a
                # view created in no_grad mode. Here it's ok to do so, so as a workaround we call conj
                # before resetting the requires_grad field for input
                input = math_op_view(input)
                assert input.is_leaf
                return input.requires_grad_(requires_grad)

            if isinstance(input, Sequence):
                out = list(map(clone_input_helper, input))
                out[0] = clone_and_perform_view(out[0])
                return tuple(out)

        for sample in samples:
            tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
            cloned1 = clone_and_perform_view(sample.input)

            # Computes function forward value with a physically conjugated/negated tensor and
            # a conj/neg view tensor and verifies that the output in both case are equal.
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)
            forward_with_mathview = op(cloned1, *sample.args, **sample.kwargs)
            self.assertEqual(expected_forward, forward_with_mathview)

            # If the op has an inplace variant, and the input doesn't require broadcasting
            # and has the same dtype as output, verify that the inplace operation on a conjugated/negated
            # input produces correct output, and the output tensor has the conj/neg bit set to True
            if inplace_variant is not None and not sample.broadcasts_input:
                cloned2 = clone_and_perform_view(tensor, requires_grad=False)
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is tensor.dtype):
                    inplace_forward = inplace_variant(cloned2, *sample.args, **sample.kwargs)
                    self.assertTrue(is_bit_set(inplace_forward))
                    self.assertEqual(inplace_forward, expected_forward)

            # TODO: backward consistency only supported for single tensor outputs
            # TODO: backward consistency only checked on sample.input, not all
            #   tensor inputs
            # TODO: update to handle checking grads of all tensor inputs as
            #   derived from each tensor output
            if isinstance(expected_forward, torch.Tensor) and expected_forward.requires_grad:
                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)
                expected_forward = output_process_fn_grad(expected_forward)
                forward_with_mathview = output_process_fn_grad(forward_with_mathview)

                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                expected_forward.sum().backward(retain_graph=True)
                forward_with_mathview.sum().backward(retain_graph=True)
                if tensor.grad is not None:
                    cloned1_tensor = cloned1 if isinstance(cloned1, torch.Tensor) else cloned1[0]
                    self.assertEqual(tensor.grad, cloned1_tensor.grad)

                    tensor.grad, cloned1_tensor.grad = None, None

                    # a repeat of the above test if output is not complex valued
                    if (out_type(expected_forward)):
                        grad = torch.randn_like(expected_forward)
                        expected_forward.backward(grad)
                        forward_with_mathview.backward(math_op_view(math_op_physical(grad)))

                        self.assertEqual(tensor.grad, cloned1_tensor.grad)

    @ops(op_db, allowed_dtypes=(torch.cfloat,))
    def test_conj_view(self, device, dtype, op):
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")
        math_op_physical = torch.conj_physical
        math_op_view = torch.conj
        _requires_grad = (op.supports_autograd and op.supports_complex_autograd(torch.device(device).type))
        is_bit_set = torch.is_conj
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, torch.is_complex)

    @ops(op_db, allowed_dtypes=(torch.double,))
    def test_neg_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        math_op_physical = torch.neg
        math_op_view = torch._neg_view
        is_bit_set = torch.is_neg
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set,
                             lambda x: True)

    @ops(op_db, allowed_dtypes=(torch.cdouble,))
    def test_neg_conj_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        def math_op_physical(x):
            return -x.conj_physical()

        def math_op_view(x):
            return torch._neg_view(x).conj()

        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)

        _requires_grad = (op.supports_autograd and op.supports_complex_autograd(torch.device(device).type))
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        # Only test one sample
        samples = itertools.islice(samples, 1)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set,
                             torch.is_complex)


instantiate_device_type_tests(TestCommon, globals())
instantiate_device_type_tests(TestCompositeCompliance, globals())
instantiate_device_type_tests(TestMathBits, globals())

if __name__ == '__main__':
    run_tests()
