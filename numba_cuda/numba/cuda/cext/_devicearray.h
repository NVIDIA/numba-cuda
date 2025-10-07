// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

#ifndef NUMBA_DEVICEARRAY_H_
#define NUMBA_DEVICEARRAY_H_

#ifdef __cplusplus
    extern "C" {
#endif

#define NUMBA_DEVICEARRAY_IMPORT_NAME "numba_cuda._devicearray"
/* These definitions should only be used by consumers of the Device Array API.
 * Consumers access the API through the opaque pointer stored in
 * _devicearray._DEVICEARRAY_API.  We don't want these definitions in
 * _devicearray.cpp itself because they would conflict with the actual
 * implementations there.
 */
#ifndef NUMBA_IN_DEVICEARRAY_CPP_

    extern void **DeviceArray_API;
    #define DeviceArrayType (*(PyTypeObject*)DeviceArray_API[0])

#endif /* ndef NUMBA_IN_DEVICEARRAY_CPP */

#ifdef __cplusplus
    }
#endif

#endif  /* NUMBA_DEVICEARRAY_H_ */
