NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -arch=sm_35

all: run_main.exe

run_main.exe: run_main.o run_kernel.o
	$(NVCC) $^ -o $@

run_main.o: main.cpp kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run_kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@