// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

/*
 * Helper functions used by Numba CUDA at runtime.
 * This C file is meant to be included after defining the
 * NUMBA_EXPORT_FUNC() and NUMBA_EXPORT_DATA() macros.
 */

#include "_pymodule.h"
#include <stddef.h>

/*
 * Unicode helpers
 */

/* Developer note:
 *
 * The hash value of unicode objects is obtained via:
 * ((PyASCIIObject *)(obj))->hash;
 * The use comes from this definition:
 * https://github.com/python/cpython/blob/6d43f6f081023b680d9db4542d19b9e382149f0a/Objects/unicodeobject.c#L119-L120
 * and it's used extensively throughout the `cpython/Object/unicodeobject.c`
 * source, not least in `unicode_hash` itself:
 * https://github.com/python/cpython/blob/6d43f6f081023b680d9db4542d19b9e382149f0a/Objects/unicodeobject.c#L11662-L11679
 *
 * The Unicode string struct layouts are described here:
 * https://github.com/python/cpython/blob/6d43f6f081023b680d9db4542d19b9e382149f0a/Include/cpython/unicodeobject.h#L82-L161
 * essentially, all the unicode string layouts start with a `PyASCIIObject` at
 * offset 0 (as of commit 6d43f6f081023b680d9db4542d19b9e382149f0a, somewhere
 * in the 3.8 development cycle).
 *
 * For safety against future CPython internal changes, the code checks that the
 * _base members of the unicode structs are what is expected in 3.7, and that
 * their offset is 0. It then walks the struct to the hash location to make sure
 * the offset is indeed the same as PyASCIIObject->hash.
 * Note: The large condition in the if should evaluate to a compile time
 * constant.
 */

#define MEMBER_SIZE(structure, member) sizeof(((structure *)0)->member)

NUMBA_EXPORT_FUNC(void *)
numba_extract_unicode(PyObject *obj, Py_ssize_t *length, int *kind,
                      unsigned int *ascii, Py_ssize_t *hash) {
    if (!PyUnicode_READY(obj)) {
        *length = PyUnicode_GET_LENGTH(obj);
        *kind = PyUnicode_KIND(obj);
        /* could also use PyUnicode_IS_ASCII but it is not publicly advertised in https://docs.python.org/3/c-api/unicode.html */
        *ascii = (unsigned int)(PyUnicode_MAX_CHAR_VALUE(obj) == (0x7f));
        /* this is here as a crude check for safe casting of all unicode string
         * structs to a PyASCIIObject */
        if (MEMBER_SIZE(PyCompactUnicodeObject, _base) == sizeof(PyASCIIObject)             &&
            MEMBER_SIZE(PyUnicodeObject, _base) == sizeof(PyCompactUnicodeObject)           &&
            offsetof(PyCompactUnicodeObject, _base) == 0                                    &&
            offsetof(PyUnicodeObject, _base) == 0                                           &&
            offsetof(PyCompactUnicodeObject, _base.hash) == offsetof(PyASCIIObject, hash)   &&
            offsetof(PyUnicodeObject, _base._base.hash) == offsetof(PyASCIIObject, hash)
           ) {
            /* Grab the hash from the type object cache, do not compute it. */
            *hash = ((PyASCIIObject *)(obj))->hash;
        }
        else {
            /* cast is not safe, fail */
            return NULL;
        }
        return PyUnicode_DATA(obj);
    } else {
        return NULL;
    }
}
