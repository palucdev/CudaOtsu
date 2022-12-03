#include "MonoCudaOtsuBinarizer.cuh"

#include <stdio.h>

// CUDA imports
#include <cuda_runtime.h>

__global__ void kernelBinarize(unsigned int* histogram, unsigned char* rawPixels, double* betweenClassVariances, double *allProbabilitySum,
	unsigned int* threshold, long totalPixels, int threadsPerBlock)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int chunkSize = ceil((float)totalPixels / (float)(threadsPerBlock));
	int startPosition = id * chunkSize;
	
	// Calculate Histogram
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			int pixelValue = (int)rawPixels[i];
			atomicAdd(&histogram[pixelValue], 1);
		}
	}

	__syncthreads();

	// Compute best class variance

	if (id == 0) {
		for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
			*allProbabilitySum += i * ((double)histogram[i] / (double)totalPixels);
			betweenClassVariances[i] = 0;
		}
	}

	__syncthreads();

	int histogramChunk = ceil((float)PngImage::MAX_PIXEL_VALUE / (float)(threadsPerBlock));

	if (id < PngImage::MAX_PIXEL_VALUE) {
		double firstClassProbability = 0, secondClassProbability = 0;
		double firstClassMean = 0, secondClassMean = 0;
		double firstProbabilitySum = 0;

		int histogramStartPosition = id * histogramChunk;
		double normalizedHistogramValue;
		for (int h = histogramStartPosition; h < (histogramStartPosition + histogramChunk); h++) {
			if (h < PngImage::MAX_PIXEL_VALUE) {
				firstClassProbability = 0;
				firstProbabilitySum = 0;
				for (int t = 0; t <= h; t++) {
					normalizedHistogramValue = ((double)histogram[t] / (double)totalPixels);
					firstClassProbability += normalizedHistogramValue;
					firstProbabilitySum += t * normalizedHistogramValue;
				}

				secondClassProbability = 1 - firstClassProbability;

				firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
				secondClassMean = (double)(*allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

				betweenClassVariances[h] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
			}
		}
	}

	__syncthreads();

	if (id == 0) {
		double maxVariance = 0;
		unsigned int currentBestThreshold = 0;
		for (int t = 0; t < PngImage::MAX_PIXEL_VALUE; t++) {
			if (betweenClassVariances[t] > maxVariance) {
				currentBestThreshold = t;
				maxVariance = betweenClassVariances[t];
			}
		}

		*threshold = currentBestThreshold;
	}

	__syncthreads();

	int bestThreshold = *threshold;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			if ((int)rawPixels[i] > bestThreshold) {
				rawPixels[i] = PngImage::COLOR_WHITE;
			}
			else {
				rawPixels[i] = PngImage::COLOR_BLACK;
			}
		}
	}
}

MonoCudaOtsuBinarizer::MonoCudaOtsuBinarizer(int threadsPerBlock, bool drawHistogram, const char* TAG) {
	this->threadsPerBlock_ = threadsPerBlock;
	this->executionTime_ = 0;

	this->showHistogram_ = drawHistogram;
	this->TAG = TAG;
}

MonoCudaOtsuBinarizer::~MonoCudaOtsuBinarizer() {}

PngImage* MonoCudaOtsuBinarizer::binarize(PngImage * imageToBinarize)
{
	long totalImagePixels = (long)imageToBinarize->getRawPixelData().size();

	unsigned char* binarizedRawPixels = cudaBinarize(imageToBinarize->getRawPixelData().data(), totalImagePixels);
	cudaDeviceSynchronize();

	std::vector<unsigned char> binarizedVector(&binarizedRawPixels[0], &binarizedRawPixels[totalImagePixels]);

	delete binarizedRawPixels;

	printf("\n\t[%s] Total calculation time: %.6f milliseconds \n", this->TAG, this->executionTime_);

	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		binarizedVector
	);
}

void MonoCudaOtsuBinarizer::showHistogram(double* histogram) {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}

unsigned char* MonoCudaOtsuBinarizer::cudaBinarize(unsigned char * rawPixels, long totalPixels) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned char* hostRawPixels = new unsigned char[totalPixels];

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	unsigned int* hostHistogram = new unsigned int[PngImage::MAX_PIXEL_VALUE];
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		hostHistogram[i] = 0;
	}

	double* hostBetweenClassVariances = new double[PngImage::MAX_PIXEL_VALUE];
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		hostBetweenClassVariances[i] = 0;
	}

	double* hostAllProbabilitySum = new double;
	*hostAllProbabilitySum = 0;

	double* deviceAllProbabilitySum;
	cudaMalloc((void **)&deviceAllProbabilitySum, sizeof(double));
	cudaMemcpy(deviceAllProbabilitySum, hostAllProbabilitySum, sizeof(double), cudaMemcpyHostToDevice);

	unsigned int hostThreshold = 0;

	unsigned int* deviceThreshold;
	cudaMalloc((void **)&deviceThreshold, sizeof(unsigned int));
	cudaMemcpy(deviceThreshold, &hostThreshold, sizeof(unsigned int), cudaMemcpyHostToDevice);

	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE);
	cudaMemcpy(deviceHistogram, hostHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

	double* deviceBetweenClassVariances;
	cudaMalloc((void **)&deviceBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE);
	cudaMemcpy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	kernelBinarize<<<1, threadsPerBlock_ >>>(deviceHistogram, deviceRawPixels, deviceBetweenClassVariances, deviceAllProbabilitySum,
		deviceThreshold, totalPixels, threadsPerBlock_);
	cudaEventRecord(stop);

	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	
	double* normalizedHistogram = new double[PngImage::MAX_PIXEL_VALUE];
	long pixelsSum = 0;
	for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}

	if (this->showHistogram_) {
		printf("\n\t[%s] Histogram pixels: %d \n", this->TAG, pixelsSum);
		showHistogram(normalizedHistogram);
	}
	
	cudaFree(deviceHistogram);
	cudaFree(deviceBetweenClassVariances);
	cudaFree(deviceAllProbabilitySum);
	cudaFree(deviceThreshold);

	cudaMemcpy(hostRawPixels, deviceRawPixels, sizeof(unsigned char) * totalPixels, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	this->executionTime_ += milliseconds;

	cudaFree(deviceRawPixels);

	delete hostHistogram;
	delete hostBetweenClassVariances;
	delete hostAllProbabilitySum;
	delete normalizedHistogram;

	return hostRawPixels;
}
