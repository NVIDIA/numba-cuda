# Generates the input files used by the pynvjitlink binding test suite

# Test binaries are built taking into account the CC of the GPU in the test machine
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | head -n 1 | sed 's/\.//')
GPU_CC ?= 75

# Use CC 7.0 as an alternative in fatbin testing, unless CC is 7.x
ifeq ($(shell echo "$(GPU_CC)" | cut -c1),7)
    ALT_CC := 80
else
    ALT_CC := 70
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

all:
	@echo "GPU CC: $(GPU_CC)"
	@echo "Alternative CC: $(ALT_CC)"
	# Compile all test objects
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $(OUTPUT_DIR)/undefined_extern.cubin undefined_extern.cu
	nvcc $(NVCC_FLAGS) $(CUBIN_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.cubin test_device_functions.cu
	nvcc $(NVCC_FLAGS) $(FATBIN_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.fatbin test_device_functions.cu
	nvcc $(NVCC_FLAGS) $(MULTI_FATBIN_FLAGS) -o $(OUTPUT_DIR)/test_device_functions_multi.fatbin test_device_functions.cu
	nvcc $(NVCC_FLAGS) $(PTX_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.ptx test_device_functions.cu
	nvcc $(NVCC_FLAGS) $(OBJECT_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.o test_device_functions.cu
	nvcc $(NVCC_FLAGS) $(LIBRARY_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.a test_device_functions.cu

	# Generate LTO-IR wrapped in a fatbin
	nvcc $(NVCC_FLAGS) $(LTOIR_FLAGS) -o $(OUTPUT_DIR)/test_device_functions.ltoir.o test_device_functions.cu
	# Generate LTO-IR in a "raw" LTO-IR container
	python generate_raw_ltoir.py --arch sm_$(GPU_CC) -o $(OUTPUT_DIR)/test_device_functions.ltoir test_device_functions.cu
