#include <stdio.h>
#include <stdlib.h>

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

__global__ void computeClassVariances(unsigned int* histogram, double allProbabilitySum, long int totalPixels, double* betweenClassVariance)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double firstProbabilitySum = 0;

	for (int t = 0; t < id; t++) {
		firstClassProbability += histogram[t];
		firstProbabilitySum += t * firstClassProbability;
	}

	secondClassProbability = totalPixels - firstClassProbability;

	firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
	secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

	betweenClassVariance[id] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
}

__global__ void binarize(unsigned char* rawPixels, long totalPixels, long chunkSize, unsigned char threshold)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			rawPixels[i] = (int)threshold > rawPixels[i] ? (unsigned char)255 : (unsigned char)0;
		}
	}
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

extern "C" unsigned char cudaFindThreshold(unsigned int* histogram, long int totalPixels) {
	int threadsPerBlock = 256;
	int numBlocks = 256;

	double allProbabilitySum = 0;
	for (int i = 0; i < 256; i++) {
		allProbabilitySum += i * histogram[i];
	}

	double* hostBetweenClassVariances = new double[256];
	for (int i = 0; i < 256; i++) {
		hostBetweenClassVariances[i] = 0;
	}

	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * 256);
	cudaMemcpy(deviceHistogram, histogram, 256 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	double* deviceBetweenClassVariances;
	cudaMalloc((void **)&deviceBetweenClassVariances, sizeof(double) * 256);
	cudaMemcpy(deviceBetweenClassVariances, hostBetweenClassVariances, 256 * sizeof(double), cudaMemcpyHostToDevice);

	computeClassVariances<<<numBlocks, threadsPerBlock>>>(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);

	cudaMemcpy(hostBetweenClassVariances, deviceBetweenClassVariances, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(deviceHistogram);
	cudaFree(deviceBetweenClassVariances);

	double maxVariance = 0;
	unsigned char currentBestThreshold = 0;
	for (int t = 0; t < 256; t++) {
		if (hostBetweenClassVariances[t] > maxVariance) {
			currentBestThreshold = t;
			maxVariance = hostBetweenClassVariances[t];
		}
	}

	return currentBestThreshold;
}

extern "C" unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold) {
	int threadsPerBlock = 256;
	int numBlocks = 256;

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	long chunkSize = totalPixels / (256 * 256);

	binarize<<<numBlocks, threadsPerBlock>>>(deviceRawPixels, totalPixels, chunkSize, threshold);

	cudaMemcpy(rawPixels, deviceRawPixels, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaFree(deviceRawPixels);

	return rawPixels;
}