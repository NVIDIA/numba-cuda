// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

/*
 * Expose all functions as pointers in a dedicated C extension.
 */

/* Import _pymodule.h first, for a recent _POSIX_C_SOURCE */
#include "_pymodule.h"

/* Visibility control macros */
#if defined(_WIN32) || defined(_WIN64)
    #define VISIBILITY_HIDDEN
    #define VISIBILITY_GLOBAL __declspec(dllexport)
#else
    #define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
    #define VISIBILITY_GLOBAL __attribute__((visibility("default")))
#endif

/* Define all runtime-required symbols in this C module, but do not
   export them outside the shared library if possible. */
#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype

/* Numba CUDA C helpers */
#include "_helperlib.c"

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &numba_##func)

    /* Unicode string support */
    declmethod(extract_unicode);

#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

static PyMethodDef ext_methods[] = {
    { NULL },
};

MOD_INIT(_helperlib) {
    PyObject *m;
    MOD_DEF(m, "_helperlib", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    PyModule_AddIntConstant(m, "long_min", LONG_MIN);
    PyModule_AddIntConstant(m, "long_max", LONG_MAX);
    PyModule_AddIntConstant(m, "py_buffer_size", sizeof(Py_buffer));
    PyModule_AddIntConstant(m, "py_gil_state_size", sizeof(PyGILState_STATE));
    PyModule_AddIntConstant(m, "py_unicode_1byte_kind", PyUnicode_1BYTE_KIND);
    PyModule_AddIntConstant(m, "py_unicode_2byte_kind", PyUnicode_2BYTE_KIND);
    PyModule_AddIntConstant(m, "py_unicode_4byte_kind", PyUnicode_4BYTE_KIND);
#if (PY_MAJOR_VERSION == 3)
#if ((PY_MINOR_VERSION == 10) || (PY_MINOR_VERSION == 11))
    PyModule_AddIntConstant(m, "py_unicode_wchar_kind", PyUnicode_WCHAR_KIND);
#endif
#endif

    return MOD_SUCCESS_VAL(m);
}
