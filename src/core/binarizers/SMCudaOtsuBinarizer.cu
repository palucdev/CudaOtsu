#include "SMCudaOtsuBinarizer.cuh"

#include <stdio.h>

// CUDA imports
#include <cuda_runtime.h>

__shared__ unsigned int myHistogram[PngImage::MAX_PIXEL_VALUE];
__global__ void smKernelCalculateHistogram(unsigned int* histogram, unsigned char* rawPixels, long chunkSize, long totalPixels, int histogramChunk)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (histogramChunk > 1) {
		int histogramStartPosition = threadIdx.x * histogramChunk;
		for (int h = histogramStartPosition; h < (histogramStartPosition + histogramChunk); h++) {
			if (h < PngImage::MAX_PIXEL_VALUE) {
				myHistogram[h] = 0;
			}
		}
	} else {
		if (threadIdx.x < PngImage::MAX_PIXEL_VALUE) { 
			myHistogram[threadIdx.x] = 0;
		}
	}

	__syncthreads();

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			int pixelValue = (int)rawPixels[i];
			atomicAdd(&myHistogram[pixelValue], 1);
		}
	}

	__syncthreads();

	if (histogramChunk > 1) {
		int histogramStartPosition = threadIdx.x * histogramChunk;
		for (int h = histogramStartPosition; h < (histogramStartPosition + histogramChunk); h++) {
			if (h < PngImage::MAX_PIXEL_VALUE) {
				atomicAdd(&histogram[h], myHistogram[h]);
			}
		}
	} else {
		if (threadIdx.x < PngImage::MAX_PIXEL_VALUE) {
			atomicAdd(&histogram[threadIdx.x], myHistogram[threadIdx.x]);
		}
	}
}

__shared__ double myHistogramCopy[PngImage::MAX_PIXEL_VALUE];
__global__ void smKernelComputeClassVariances(double* histogram, double allProbabilitySum, long int totalPixels, double* betweenClassVariance)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	myHistogramCopy[threadIdx.x % PngImage::MAX_PIXEL_VALUE] = histogram[threadIdx.x % PngImage::MAX_PIXEL_VALUE];
	
	__syncthreads();

	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double firstProbabilitySum = 0;

	for (int t = 0; t <= id % PngImage::MAX_PIXEL_VALUE; t++) {
		firstClassProbability += myHistogramCopy[t];
		firstProbabilitySum += t * myHistogramCopy[t];
	}

	secondClassProbability = 1 - firstClassProbability;

	firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
	secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

	betweenClassVariance[id] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
}

SMCudaOtsuBinarizer::SMCudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram) : CudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistogram, "GPU - Shared Memory") {
	this->threadsPerBlock_ = threadsPerBlock;
	this->numBlocks_ = numBlocks;
	this->drawHistogram_ = drawHistogram;
}

SMCudaOtsuBinarizer::~SMCudaOtsuBinarizer() {}

unsigned char SMCudaOtsuBinarizer::cudaFindThreshold(double* histogram, long int totalPixels) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int threadsPerBlock = 256;
	int numBlocks = 1;

	double allProbabilitySum = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		allProbabilitySum += i * histogram[i];
	}

	double* hostBetweenClassVariances = new double[PngImage::MAX_PIXEL_VALUE];
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		hostBetweenClassVariances[i] = 0;
	}

	double* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(double) * PngImage::MAX_PIXEL_VALUE);
	cudaMemcpy(deviceHistogram, histogram, sizeof(double) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

	double* deviceBetweenClassVariances;
	cudaMalloc((void **)&deviceBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE);
	cudaMemcpy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	smKernelComputeClassVariances << <numBlocks, threadsPerBlock >> >(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);
	cudaEventRecord(stop);
	cudaMemcpy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\t[%s] Threshold calculated in %.6f milliseconds \n", this->TAG, milliseconds);
	binarizerTimestamp_->thresholdFindingTimeInSeconds += milliseconds / 1000;

	cudaFree(deviceHistogram);
	cudaFree(deviceBetweenClassVariances);

	double maxVariance = 0;
	unsigned char currentBestThreshold = 0;
	for (int t = 0; t < PngImage::MAX_PIXEL_VALUE; t++) {
		if (hostBetweenClassVariances[t] > maxVariance) {
			currentBestThreshold = (unsigned char)t;
			maxVariance = hostBetweenClassVariances[t];
		}
	}

	delete hostBetweenClassVariances;

	return currentBestThreshold;
}

double* SMCudaOtsuBinarizer::cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels) {
	//TODO: check cudaGetDeviceProperties function!

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned int* hostHistogram = new unsigned int[PngImage::MAX_PIXEL_VALUE];
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		hostHistogram[i] = 0;
	}

	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE);
	cudaMemcpy(deviceHistogram, hostHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, sizeof(unsigned char) * totalPixels, cudaMemcpyHostToDevice);

	long chunkSize = ceil(totalPixels / (threadsPerBlock_ * numBlocks_)) + 1;

	int histogramChunk = ceil(PngImage::MAX_PIXEL_VALUE / threadsPerBlock_) + 1;

	cudaEventRecord(start);
	smKernelCalculateHistogram<<<numBlocks_, threadsPerBlock_ >>>(deviceHistogram, deviceRawPixels, chunkSize, totalPixels, histogramChunk);
	cudaEventRecord(stop);

	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\t[%s] Histogram calculated in %.6f milliseconds \n", this->TAG, milliseconds);
	binarizerTimestamp_->histogramBuildingTimeInSeconds += milliseconds / 1000;

	cudaFree(deviceHistogram);
	cudaFree(deviceRawPixels);

	double* normalizedHistogram = new double[PngImage::MAX_PIXEL_VALUE];
	long pixelsSum = 0;
	for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}
	printf("\n\t[%s] Histogram pixels: %d \n", this->TAG, pixelsSum);

	delete hostHistogram;

	return normalizedHistogram;
}
