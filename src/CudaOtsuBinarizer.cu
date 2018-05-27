#include "CudaOtsuBinarizer.cuh"

// CUDA imports
#include <cuda_runtime.h>

__global__ void kernelCalculateHistogram(unsigned int* histogram, unsigned char* rawPixels, long chunkSize, long totalPixels)
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

__global__ void kernelComputeClassVariances(double* histogram, double allProbabilitySum, long int totalPixels, double* betweenClassVariance)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double firstProbabilitySum = 0;

	for (int t = 0; t <= id; t++) {
		firstClassProbability += histogram[t];
		firstProbabilitySum += t * histogram[t];
	}

	secondClassProbability = 1 - firstClassProbability;

	firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
	secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

	betweenClassVariance[id] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
}

__global__ void kernelBinarize(unsigned char* rawPixels, long totalPixels, long chunkSize, unsigned char threshold)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			if ((int)rawPixels[i] >(int)threshold) {
				rawPixels[i] = PngImage::COLOR_WHITE;
			}
			else {
				rawPixels[i] = PngImage::COLOR_BLACK;
			}
		}
	}
}

CudaOtsuBinarizer::CudaOtsuBinarizer() {}

PngImage* CudaOtsuBinarizer::binarize(PngImage * imageToBinarize)
{
	long totalImagePixels = (long)imageToBinarize->getRawPixelData().size();

	double* histogram = cudaCalculateHistogram(imageToBinarize->getRawPixelData().data(), totalImagePixels);
	cudaDeviceSynchronize();
	showHistogram(histogram);

	unsigned char threshold;
	threshold = cudaFindThreshold(histogram, totalImagePixels);
	cudaDeviceSynchronize();
	printf("\t[GPU] Threshold value: %d\n", threshold);

	delete histogram;

	unsigned char* binarizedRawPixels = cudaBinarize(imageToBinarize->getRawPixelData().data(), totalImagePixels, threshold);
	cudaDeviceSynchronize();

	std::vector<unsigned char> binarizedVector(&binarizedRawPixels[0], &binarizedRawPixels[totalImagePixels]);

	delete binarizedRawPixels;

	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		binarizedVector
	);
}

void CudaOtsuBinarizer::showHistogram(double* histogram) {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}

double* CudaOtsuBinarizer::cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels) {
	int threadsPerBlock = 256;
	int numBlocks = 256;

	//TODO: check cudaGetDeviceProperties function!

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

	long chunkSize = ceil(totalPixels / (threadsPerBlock * numBlocks)) + 1;

	kernelCalculateHistogram << <numBlocks, threadsPerBlock >> >(deviceHistogram, deviceRawPixels, chunkSize, totalPixels);
	
	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

	cudaFree(deviceHistogram);
	cudaFree(deviceRawPixels);

	double* normalizedHistogram = new double[PngImage::MAX_PIXEL_VALUE];
	for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
	}

	free(hostHistogram);

	return normalizedHistogram;
}

unsigned char CudaOtsuBinarizer::cudaFindThreshold(double* histogram, long int totalPixels) {
	int threadsPerBlock = 16;
	int numBlocks = 16;

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

	delete hostBetweenClassVariances;

	kernelComputeClassVariances<<<numBlocks, threadsPerBlock>>>(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);
	cudaMemcpy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * PngImage::MAX_PIXEL_VALUE, cudaMemcpyDeviceToHost);

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

	return currentBestThreshold;
}

unsigned char* CudaOtsuBinarizer::cudaBinarize(unsigned char * rawPixels, long totalPixels, unsigned char threshold) {
	int threadsPerBlock = 256;
	int numBlocks = 256;

	unsigned char* hostRawPixels = new unsigned char[totalPixels];

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	long chunkSize = ceil(totalPixels / (threadsPerBlock * numBlocks)) + 1;

	kernelBinarize<<<numBlocks, threadsPerBlock>>>(deviceRawPixels, totalPixels, chunkSize, threshold);

	cudaMemcpy(hostRawPixels, deviceRawPixels, sizeof(unsigned char) * totalPixels, cudaMemcpyDeviceToHost);

	cudaFree(deviceRawPixels);

	return hostRawPixels;
}
