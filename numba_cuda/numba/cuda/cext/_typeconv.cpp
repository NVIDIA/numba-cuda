// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

#include "_pymodule.h"
#include "capsulethunk.h"
#include "typeconv.hpp"

extern "C" {


static PyObject*
new_type_manager(PyObject* self, PyObject* args);

static void
del_type_manager(PyObject *);

static PyObject*
select_overload(PyObject* self, PyObject* args);

static PyObject*
check_compatible(PyObject* self, PyObject* args);

static PyObject*
set_compatible(PyObject* self, PyObject* args);

static PyObject*
get_pointer(PyObject* self, PyObject* args);


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(new_type_manager),
    declmethod(select_overload),
    declmethod(check_compatible),
    declmethod(set_compatible),
    declmethod(get_pointer),
    { NULL },
#undef declmethod
};


MOD_INIT(_typeconv) {
    PyObject *m;
    MOD_DEF(m, "_typeconv", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}

} // end extern C

///////////////////////////////////////////////////////////////////////////////

const char PY_CAPSULE_TM_NAME[] = "*tm";
#define BAD_TM_ARGUMENT PyErr_SetString(PyExc_TypeError,                    \
                                        "1st argument not TypeManager")

static
TypeManager* unwrap_TypeManager(PyObject *tm) {
    void* p = PyCapsule_GetPointer(tm, PY_CAPSULE_TM_NAME);
    return reinterpret_cast<TypeManager*>(p);
}

PyObject*
new_type_manager(PyObject* self, PyObject* args)
{
    TypeManager* tm = new TypeManager();
    return PyCapsule_New(tm, PY_CAPSULE_TM_NAME, &del_type_manager);
}

void
del_type_manager(PyObject *tm)
{
    delete unwrap_TypeManager(tm);
}

PyObject*
select_overload(PyObject* self, PyObject* args)
{
    PyObject *tmcap, *sigtup, *ovsigstup;
    int allow_unsafe;
    int exact_match_required;
    PyObject *sigtup_fast = NULL;
    PyObject *ovsigstup_fast = NULL;
    PyObject *cursig_fast = NULL;
    Type *sig = NULL;
    Type *ovsigs = NULL;
    PyObject *result = NULL;
    Py_ssize_t sigsz = 0;
    Py_ssize_t ovsz = 0;

    if (!PyArg_ParseTuple(args, "OOOii", &tmcap, &sigtup, &ovsigstup,
                          &allow_unsafe, &exact_match_required)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if (!tm) {
        BAD_TM_ARGUMENT;
    }

    sigtup_fast = PySequence_Fast(sigtup, "1st argument should be a sequence");
    if (!sigtup_fast) {
        goto done;
    }
    ovsigstup_fast = PySequence_Fast(ovsigstup,
                                     "2nd argument should be a sequence");
    if (!ovsigstup_fast) {
        goto done;
    }

    sigsz = PySequence_Fast_GET_SIZE(sigtup_fast);
    ovsz = PySequence_Fast_GET_SIZE(ovsigstup_fast);

    if (ovsz != 0 && sigsz > PY_SSIZE_T_MAX / ovsz) {
        PyErr_SetString(PyExc_OverflowError, "Too many overload signatures");
        goto done;
    }

    if (sigsz < 0 || ovsz < 0) {
        goto done;
    }

    sig = new Type[sigsz];
    ovsigs = new Type[ovsz * sigsz];

    for (Py_ssize_t i = 0; i < sigsz; ++i) {
        long tid = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(sigtup_fast, i),
                                      NULL);
        if (tid == -1 && PyErr_Occurred()) {
            goto done;
        }
        sig[i] = Type(tid);
    }

    for (Py_ssize_t i = 0; i < ovsz; ++i) {
        cursig_fast = PySequence_Fast(PySequence_Fast_GET_ITEM(ovsigstup_fast, i),
                                     "Each overload should be a sequence");
        if (!cursig_fast) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(cursig_fast) != sigsz) {
            PyErr_SetString(
                PyExc_TypeError,
                "Each overload should have same length as provided signature"
            );
            Py_DECREF(cursig_fast);
            cursig_fast = NULL;
            goto done;
        }
        for (Py_ssize_t j = 0; j < sigsz; ++j) {
            long tid = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(cursig_fast, j),
                                          NULL);
            if (tid == -1 && PyErr_Occurred()) {
                goto done;
            }
            ovsigs[i * sigsz + j] = Type(tid);
        }
        Py_DECREF(cursig_fast);
        cursig_fast = NULL;
    }

    int selected = -42;
    int matches = tm->selectOverload(sig, ovsigs, selected, sigsz, ovsz,
                                     (bool) allow_unsafe,
                                     (bool) exact_match_required);

    if (matches > 1) {
        PyErr_SetString(PyExc_TypeError, "Ambiguous overloading");
        goto done;
    } else if (matches == 0) {
        PyErr_SetString(PyExc_TypeError, "No compatible overload");
        goto done;
    }

    result = PyLong_FromLong(selected);

done:
    delete [] sig;
    delete [] ovsigs;
    Py_XDECREF(sigtup_fast);
    Py_XDECREF(ovsigstup_fast);
    Py_XDECREF(cursig_fast);

    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        return NULL;
    }

    return result;
}

PyObject*
check_compatible(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    int from, to;
    if (!PyArg_ParseTuple(args, "Oii", &tmcap, &from, &to)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if(!tm) {
        BAD_TM_ARGUMENT;
        return NULL;
    }

    switch(tm->isCompatible(Type(from), Type(to))){
    case TCC_EXACT:
        return PyString_FromString("exact");
    case TCC_PROMOTE:
        return PyString_FromString("promote");
    case TCC_CONVERT_SAFE:
        return PyString_FromString("safe");
        case TCC_CONVERT_UNSAFE:
        return PyString_FromString("unsafe");
    default:
        Py_RETURN_NONE;
    }
}

PyObject*
set_compatible(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    int from, to, by;
    if (!PyArg_ParseTuple(args, "Oiii", &tmcap, &from, &to, &by)) {
        return NULL;
    }

    TypeManager *tm = unwrap_TypeManager(tmcap);
    if (!tm) {
        BAD_TM_ARGUMENT;
        return NULL;
    }
    TypeCompatibleCode tcc;
    switch (by) {
    case 'p': // promote
        tcc = TCC_PROMOTE;
        break;
    case 's': // safe convert
        tcc = TCC_CONVERT_SAFE;
        break;
    case 'u': // unsafe convert
        tcc = TCC_CONVERT_UNSAFE;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unknown TCC");
        return NULL;
    }

    tm->addCompatibility(Type(from), Type(to), tcc);
    Py_RETURN_NONE;
}


PyObject*
get_pointer(PyObject* self, PyObject* args)
{
    PyObject *tmcap;
    if (!PyArg_ParseTuple(args, "O", &tmcap)) {
        return NULL;
    }
    return PyLong_FromVoidPtr(unwrap_TypeManager(tmcap));
}
