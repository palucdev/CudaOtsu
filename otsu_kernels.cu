#include <stdio.h>

// CUDA Runtime
#include <cuda_runtime.h>
#define HISTOGRAM_SIZE 255

__global__ void calculateHistogram(unsigned int* histogram, unsigned char* rawPixels, long chunkSize, long totalPixels)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			int pixelValue = (int)rawPixels[i];
			atomicAdd(&histogram[pixelValue], 1);
		}
	}

}

__global__ void binarize()
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
}

extern "C" unsigned int* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels) {
	int threadsPerBlock = 256;
	int numBlocks = 256;

	//TODO: check cudaGetDeviceProperties function!

	unsigned int* hostHistogram = new unsigned int[256];
	for (int i = 0; i < 256; i++) {
		hostHistogram[i] = 0;
	}

	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int)*256);
	cudaMemcpy(deviceHistogram, hostHistogram, 256 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char)*totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	long chunkSize = totalPixels / (256 * 256);

	calculateHistogram<<<numBlocks, threadsPerBlock>>>(deviceHistogram, deviceRawPixels, chunkSize, totalPixels);

	cudaMemcpy(hostHistogram, deviceHistogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(deviceHistogram);
	cudaFree(deviceRawPixels);

	return hostHistogram;
}

extern "C" void cudaBinarize() {
	dim3 dimGrid(2, 2);
	dim3 dimBlock(2, 2, 2);

	binarize<<<dimGrid, dimBlock>>>();
}

