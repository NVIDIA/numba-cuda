# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np

import unittest
from numba.cuda import jit, grid, to_device, device_array
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


a0 = np.array(42)

s1 = np.int32(64)

a1 = np.arange(12)
a2 = a1[::2]
a3 = a1.reshape((3, 4)).T

dt = np.dtype([("x", np.int8), ("y", "S3")])

a4 = np.arange(32, dtype=np.int8).view(dt)
a5 = a4[::-2]

# A recognizable data string
a6 = np.frombuffer(b"XXXX_array_contents_XXXX", dtype=np.float32)


myarray = np.array(
    [
        1,
    ]
)


@jit
def getitem0_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[()]


@jit
def getitem1_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[i]


@jit
def getitem2_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[i]


@jit
def getitem3_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        # For 2D arrays, flatten the indexing
        if input_array.ndim > 1:
            # Calculate 2D indices from flat index
            flat_idx = i
            if flat_idx < input_array.size:
                row = flat_idx // input_array.shape[1]
                col = flat_idx % input_array.shape[1]
                output_array[i] = input_array[row, col]
        else:
            output_array[i] = input_array[i]


@jit
def getitem4_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[i]


@jit
def getitem5_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[i]


@jit
def getitem6_kernel(input_array, output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = input_array[i]


@jit
def use_arrayscalar_const_kernel(output_array, size):
    i = grid(1)
    if i < size:
        output_array[i] = s1


@jit
def write_to_global_array_kernel(global_array):
    i = grid(1)
    if i < 1:
        global_array[0] = 1


@jit
def bytes_as_const_array_kernel(output_array, size):
    i = grid(1)
    if i < size:
        # Use hardcoded bytes values instead of frombuffer
        # "foo" as uint8 values: f=102, o=111, o=111
        if i % 3 == 0:
            output_array[i] = 102  # 'f'
        elif i % 3 == 1:
            output_array[i] = 111  # 'o'
        else:
            output_array[i] = 111  # 'o'


@skip_on_cudasim("CUDA simulator does not support array constants")
class TestConstantArray(CUDATestCase):
    """
    Test array constants.
    """

    def check_array_const(self, kernel_func, input_array, expected_size):
        # Convert input array to device (make contiguous if needed)
        if (
            not input_array.flags["C_CONTIGUOUS"]
            and not input_array.flags["F_CONTIGUOUS"]
        ):
            input_array = np.ascontiguousarray(input_array)
        d_input = to_device(input_array)
        d_output = device_array(expected_size, dtype=input_array.dtype)

        # Launch kernel
        kernel_func[1, expected_size](d_input, d_output, expected_size)

        # Get result
        result = d_output.copy_to_host()

        # Verify result matches expected values
        for i in range(expected_size):
            if input_array.ndim == 0:
                # For 0D arrays, all results should be the same
                expected_val = input_array[()]
                np.testing.assert_array_equal(result[i], expected_val)
            elif i < input_array.size:
                if input_array.ndim == 1:
                    expected_val = input_array[i]
                else:
                    # For multi-dimensional arrays, flatten the indexing
                    flat_idx = i
                    if flat_idx < input_array.size:
                        expected_val = input_array.flat[flat_idx]
                    else:
                        continue
                np.testing.assert_array_equal(result[i], expected_val)

    def test_array_const_0d(self):
        self.check_array_const(getitem0_kernel, a0, 3)

    def test_array_const_1d_contig(self):
        self.check_array_const(getitem1_kernel, a1, 3)

    def test_array_const_1d_noncontig(self):
        self.check_array_const(getitem2_kernel, a2, 3)

    def test_array_const_2d(self):
        self.check_array_const(getitem3_kernel, a3, 3)

    def test_record_array_const_contig(self):
        self.check_array_const(getitem4_kernel, a4, 3)

    def test_record_array_const_noncontig(self):
        self.check_array_const(getitem5_kernel, a5, 3)

    def test_array_const_alignment(self):
        """
        Issue #1933: the array declaration in the LLVM IR must have
        the right alignment specified.
        """
        # Test the kernel with the alignment array
        self.check_array_const(getitem6_kernel, a6, 3)

    def test_arrayscalar_const(self):
        # Test arrayscalar constant in CUDA kernel
        d_output = device_array(1, dtype=np.int32)
        use_arrayscalar_const_kernel[1, 1](d_output, 1)
        result = d_output.copy_to_host()
        self.assertEqual(result[0], s1)

    def test_write_to_global_array(self):
        # Test that writing to global array works in CUDA
        d_myarray = to_device(myarray.copy())  # Make a writable copy
        write_to_global_array_kernel[1, 1](d_myarray)
        # Copy back to host and verify the global array was modified
        result = d_myarray.copy_to_host()
        self.assertEqual(result[0], 1)

    def test_issue_1850(self):
        """
        This issue is caused by an unresolved bug in numpy since version 1.6.
        See numpy GH issue #3147.
        """
        constarr = np.array([86])

        @jit
        def issue_1850_kernel(output_array, size):
            i = grid(1)
            if i < size:
                output_array[i] = constarr[0]

        d_output = device_array(1, dtype=np.int32)
        issue_1850_kernel[1, 1](d_output, 1)
        out = d_output.copy_to_host()
        self.assertEqual(out[0], 86)

    def test_too_big_to_freeze(self):
        """
        Test issue https://github.com/numba/numba/issues/2188 where freezing
        a constant array into the code that's prohibitively long and consumes
        too much RAM.
        """
        nelem = 10**4  # Reduced size for CUDA testing

        @jit
        def big_array_kernel(input_array, output_array, size):
            i = grid(1)
            if i < size:
                if input_array.ndim == 1:
                    output_array[i] = input_array[i]
                else:
                    # For multi-dimensional arrays, flatten the indexing
                    flat_idx = i
                    if flat_idx < input_array.size:
                        output_array[i] = input_array.flat[flat_idx]

        c_array = np.arange(nelem).reshape(nelem)
        f_array = np.asfortranarray(np.random.random((2, nelem // 2)))

        # Test C contig
        d_input = to_device(c_array)
        d_output = device_array(nelem, dtype=c_array.dtype)
        # Use proper block size for CUDA (max 1024 threads per block)
        block_size = min(nelem, 1024)
        grid_size = (nelem + block_size - 1) // block_size
        big_array_kernel[grid_size, block_size](d_input, d_output, nelem)
        result = d_output.copy_to_host()
        np.testing.assert_array_equal(c_array, result)

        # Test F contig
        d_input = to_device(f_array)
        d_output = device_array(f_array.size, dtype=f_array.dtype)
        block_size = min(f_array.size, 1024)
        grid_size = (f_array.size + block_size - 1) // block_size
        big_array_kernel[grid_size, block_size](d_input, d_output, f_array.size)
        result = d_output.copy_to_host()
        np.testing.assert_array_equal(f_array.flatten(), result)


@skip_on_cudasim("CUDA simulator does not support array constants")
class TestConstantBytes(CUDATestCase):
    def test_constant_bytes(self):
        # Test constant bytes array in CUDA kernel
        d_output = device_array(3, dtype=np.uint8)
        bytes_as_const_array_kernel[1, 3](d_output, 3)
        result = d_output.copy_to_host()
        expected = np.frombuffer(b"foo", dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
