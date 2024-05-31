/*
 * Helper binding to call some CUDA Runtime API that cannot be directly
 * encoded using ctypes.
 */

#include "Python.h"

#define CUDA_IPC_HANDLE_SIZE 64

typedef int CUresult;
typedef void* CUdeviceptr;

typedef struct CUipcMemHandle_st{
    char reserved[CUDA_IPC_HANDLE_SIZE];
} CUipcMemHandle;

typedef CUresult (*cuIpcOpenMemHandle_t)(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags );

static
cuIpcOpenMemHandle_t cuIpcOpenMemHandle = 0;

static
void set_cuIpcOpenMemHandle(void* fnptr)
{
    cuIpcOpenMemHandle = (cuIpcOpenMemHandle_t)fnptr;
}

static
CUresult call_cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle* handle, unsigned int flags)
{
    return cuIpcOpenMemHandle(pdptr, *handle, flags);
}


PyMODINIT_FUNC PyInit__extras(void) {
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "_extras", "No docs", -1, NULL, NULL, NULL, NULL, NULL
    };

    PyObject *m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

    PyModule_AddObject(m, "set_cuIpcOpenMemHandle", PyLong_FromVoidPtr(&set_cuIpcOpenMemHandle));
    PyModule_AddObject(m, "call_cuIpcOpenMemHandle", PyLong_FromVoidPtr(&call_cuIpcOpenMemHandle));
    PyModule_AddIntConstant(m, "CUDA_IPC_HANDLE_SIZE", CUDA_IPC_HANDLE_SIZE);
    return m;
}
