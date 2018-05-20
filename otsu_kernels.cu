#include <stdio.h>

// CUDA Runtime
#include <cuda_runtime.h>

__global__ void calculateHistogram()
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("Id value: %d\n", id);
}

__global__ void binarize()
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	printf("Id value: %d\n", id);
}

extern "C" void cudaCalculateHistogram() {
	dim3 dimGrid(2, 2);
	dim3 dimBlock(2, 2, 2);

	calculateHistogram<<<dimGrid, dimBlock>>>();
}

extern "C" void cudaBinarize() {
	dim3 dimGrid(2, 2);
	dim3 dimBlock(2, 2, 2);

	binarize<<<dimGrid, dimBlock>>>();
}

