#!/usr/bin/env python3

import numpy as np
from numba import cuda

num_entries = 8

#
# In the following code, we use explicit breakpoint function calls to insert
# breakpoints.
# 
# Breakpoints can also be set by clicking to the left of the source line numbers in the
# Editor window, and are shown as red dots next to the line numbers. The can also be set
# from the CUDA GDB command line using the filename:line-number syntax, or by kernel or function name.
#
@cuda.jit(debug=True, opt=False)
def hello(input, output):
    gid = cuda.grid(1)
    size = len(input)
    if gid >= size:
        return

    # Print the value of the array at the current thread's index
    # Output will likely not happen until the kernel is finished executing
    # because host synchronization is needed to print the output.
    # Output is not guaranteed to be printed in thread order.
    print('input[', gid, '] =', input[gid])

    # Reverse the input array
    output[gid] = input[size - gid - 1]

    # Synchronize all threads in the block
    cuda.syncthreads()

    # Have the first thread print a message to indicate that all threads
    # have synchronized
    if gid == 0:
        print('All threads have synchronized (local memory array)')

    # Print the value of the output array at the current thread's index
    print('output[', gid, '] =', output[gid])

    # Allocate a new array in shared memory
    shared_array = cuda.shared.array(num_entries, dtype=np.int64)

    # Hit a manually-inserted breakpoint here
    breakpoint()

    # Fill the shared array with the input array with negative values in reverse order
    shared_array[gid] = - input[size - gid - 1]

    # Synchronize all threads in the block
    cuda.syncthreads()

    # Have the first thread print a message to indicate that all threads
    # have synchronized
    if gid == 0:
        print('All threads have synchronized (shared memory array)')

    # Print the value of the shared array at the current thread's index
    print('shared_array[', gid, '] =', shared_array[gid])

    # Demonstrate polymorphic variables by setting the value of a variable to a different type.
    # The print() calls are used to ensure that the variable is not optimized out.
    # Stepping through the code will show how the variable changes value and type with each assignment.
    # Only let the first thread do this to reduce clutter in the output.
    if gid == 0:
        # Hit another manually-inserted breakpoint here
        breakpoint()

        # Set the variable to different values and types.
        variable = True
        print('variable =', variable)
        variable = 0x80000000
        print('variable =', variable)
        variable = 0x8000000000000000
        print('variable =', variable)
        variable = 3.141592653589793
        print('variable =', variable)
        variable = 2.718281828459045
        print('variable =', variable)

if __name__ == '__main__':
    # Generate data
    input = cuda.to_device(np.array(range(num_entries), dtype=np.int64))
    print(f'input: {input.copy_to_host()}')
 
    # Create a vector to hold the results (same size as the input)
    output = cuda.to_device(np.zeros(len(input), dtype=np.int64))

    # Launch the kernel
    hello[1, len(input)](input, output)

    # Print the results
    print(f'output: {output.copy_to_host()}')
    print('All Done!')
