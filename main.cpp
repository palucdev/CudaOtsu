// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA Runtime
#include <cuda_runtime.h>

extern "C" void cudaCalculateHistogram();
extern "C" void cudaBinarize();

int main(int argc, char **argv)
{
	//Kernel configuration, where a two-dimensional grid and
	//three-dimensional blocks are configured.
	
	cudaCalculateHistogram();
	cudaDeviceSynchronize();
	cudaBinarize();
	cudaDeviceSynchronize();

	system("pause");

	return 0;
}