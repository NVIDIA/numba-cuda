# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
.PHONY: all nrt_extern test_device_functions clean

.DEFAULT_GOAL := all

# Generates the input files used by the nvjitlink tests

# Test binaries are built taking into account the CC of the GPU in the test machine
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | head -n 1 | sed 's/\.//')
GPU_CC ?= 75

# Use CC 7.5 as an alternative in fatbin testing, unless CC is 7.x
ifeq ($(shell echo "$(GPU_CC)" | cut -c1),7)
    ALT_CC := 80
else
    ALT_CC := 75
endif

# Gencode flags suitable for most tests
GENCODE := -gencode arch=compute_$(GPU_CC),code=sm_$(GPU_CC)

MULTI_GENCODE := -gencode arch=compute_$(GPU_CC),code=[sm_$(GPU_CC),lto_$(GPU_CC)]

# Fatbin tests need to generate code for an additional compute capability
FATBIN_GENCODE := $(GENCODE) -gencode arch=compute_$(ALT_CC),code=sm_$(ALT_CC)

# Fatbin that contains both LTO, SASS for multiple architectures
MULTI_FATBIN_GENCODE := $(MULTI_GENCODE) -gencode arch=compute_$(ALT_CC),code=[sm_$(ALT_CC),lto_$(ALT_CC)]

# LTO-IR tests need to generate for the LTO "architecture" instead
LTOIR_GENCODE := -gencode arch=lto_$(GPU_CC),code=lto_$(GPU_CC)

# Compile with optimization; use relocatable device code to preserve device
# functions in the final output
NVCC_FLAGS := -O3 -rdc true

# Flags specific to output type
CUBIN_FLAGS := $(GENCODE) --cubin
PTX_FLAGS := $(GENCODE) -ptx
OBJECT_FLAGS := $(GENCODE) -dc
LIBRARY_FLAGS := $(GENCODE) -lib
FATBIN_FLAGS := $(FATBIN_GENCODE) --fatbin
MULTI_FATBIN_FLAGS := $(MULTI_FATBIN_GENCODE) --fatbin
LTOIR_FLAGS := $(LTOIR_GENCODE) -dc

OUTPUT_DIR := ./

NRT_INCLUDE_DIR := $(shell python -c "from numba.cuda.memory_management.nrt import get_include; print(get_include())")

$(OUTPUT_DIR)/undefined_extern.cubin:
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $@ undefined_extern.cu

$(OUTPUT_DIR)/test_device_functions.cubin:
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.fatbin:
	nvcc $(NVCC_FLAGS) $(FATBIN_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions_multi.fatbin:
	nvcc $(NVCC_FLAGS) $(MULTI_FATBIN_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.ptx:
	nvcc $(NVCC_FLAGS) $(PTX_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.o:
	nvcc $(NVCC_FLAGS) $(OBJECT_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.a:
	nvcc $(NVCC_FLAGS) $(LIBRARY_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.ltoir.o:
	nvcc $(NVCC_FLAGS) $(LTOIR_FLAGS) -o $@ test_device_functions.cu

$(OUTPUT_DIR)/test_device_functions.ltoir:
	# Generate LTO-IR in a "raw" LTO-IR container
	python generate_raw_ltoir.py --arch sm_$(GPU_CC) -o $(OUTPUT_DIR)/test_device_functions.ltoir test_device_functions.cu

test_device_functions: $(OUTPUT_DIR)/test_device_functions.cubin \
    $(OUTPUT_DIR)/test_device_functions.fatbin \
    $(OUTPUT_DIR)/test_device_functions_multi.fatbin \
    $(OUTPUT_DIR)/test_device_functions.ptx \
    $(OUTPUT_DIR)/test_device_functions.o \
    $(OUTPUT_DIR)/test_device_functions.a \
    $(OUTPUT_DIR)/test_device_functions.ltoir.o \
    $(OUTPUT_DIR)/test_device_functions.ltoir

$(OUTPUT_DIR)/nrt_extern.cubin:
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.fatbin:
	nvcc $(NVCC_FLAGS) $(FATBIN_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern_multi.fatbin:
	nvcc $(NVCC_FLAGS) $(MULTI_FATBIN_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.ptx:
	nvcc $(NVCC_FLAGS) $(PTX_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.o:
	nvcc $(NVCC_FLAGS) $(OBJECT_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.a:
	nvcc $(NVCC_FLAGS) $(LIBRARY_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.ltoir.o:
	nvcc $(NVCC_FLAGS) $(LTOIR_FLAGS) -o $@ nrt_extern.cu -I$(NRT_INCLUDE_DIR)

$(OUTPUT_DIR)/nrt_extern.ltoir:
	python generate_raw_ltoir.py --arch sm_$(GPU_CC) -o $@ nrt_extern.cu --nrt

nrt_extern: $(OUTPUT_DIR)/nrt_extern.cubin \
    $(OUTPUT_DIR)/nrt_extern.fatbin \
    $(OUTPUT_DIR)/nrt_extern_multi.fatbin \
    $(OUTPUT_DIR)/nrt_extern.ptx \
    $(OUTPUT_DIR)/nrt_extern.o \
    $(OUTPUT_DIR)/nrt_extern.a \
    $(OUTPUT_DIR)/nrt_extern.ltoir.o \
    $(OUTPUT_DIR)/nrt_extern.ltoir

all: test_device_functions nrt_extern $(OUTPUT_DIR)/undefined_extern.cubin
	@echo "GPU CC: $(GPU_CC)"
	@echo "Alternative CC: $(ALT_CC)"

clean:
	rm -f *.cubin *.fatbin *.ptx *.o *.a *.ltoir
