#include "OtsuBinarizer.h"

// CUDA imports
#include <cuda_runtime.h>

extern "C" unsigned int* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
extern "C" unsigned char cudaFindThreshold(unsigned int* histogram, long int totalPixels);
extern "C" unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);

OtsuBinarizer::OtsuBinarizer(){}

PngImage* OtsuBinarizer::binarizeOnCpu(PngImage * imageToBinarize)
{
	std::vector<unsigned int> histogram(PngImage::MAX_PIXEL_VALUE);

	calculateHistogram(imageToBinarize->getRawPixelData(), histogram);

	showHistogram(histogram.data());

	int threshold = findThreshold(histogram, imageToBinarize->getTotalPixels());

	printf("\t[CPU] Threshold value: %d", threshold);

	return binarizeImage(imageToBinarize, threshold);
}

void OtsuBinarizer::calculateHistogram(std::vector<unsigned char>& image, std::vector<unsigned int>& histogram) {
	std::vector<unsigned char> occurences(PngImage::MAX_PIXEL_VALUE);
	unsigned char pixelValue;

	for (std::vector<unsigned char>::size_type i = 0; i != image.size(); i++) {
		pixelValue = image[i];
		histogram[pixelValue]++;
	}
}

int OtsuBinarizer::findThreshold(std::vector<unsigned int>& histogram, long int totalPixels) {
	int threshold;
	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double betweenClassVariance = 0, maxVariance = 0;
	double allProbabilitySum = 0, firstProbabilitySum = 0;

	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		allProbabilitySum += i * histogram[i];
	}

	for (int t = 0; t < PngImage::MAX_PIXEL_VALUE; t++) {
		firstClassProbability += histogram[t];
		secondClassProbability = totalPixels - firstClassProbability;

		firstProbabilitySum += t * histogram[t];
		firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
		secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

		betweenClassVariance = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);

		if (betweenClassVariance > maxVariance) {
			threshold = t;
			maxVariance = betweenClassVariance;
		}
	}

	return threshold;
}

PngImage* OtsuBinarizer::binarizeImage(PngImage* imageToBinarize, int threshold) {
	std::vector<unsigned char> imagePixels = imageToBinarize->getRawPixelData();
	for (std::vector<unsigned char>::size_type i = 0; i != imageToBinarize->getTotalPixels(); i++) {
		if ((int)imagePixels[i] > threshold) {
			imagePixels[i] = PngImage::COLOR_WHITE;
		} else {
			imagePixels[i] = PngImage::COLOR_BLACK;
		}
	}

	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		imagePixels
	);
}

PngImage* OtsuBinarizer::binarizeOnGpu(PngImage * imageToBinarize)
{
	unsigned int* histogram = new unsigned int[PngImage::MAX_PIXEL_VALUE];
	long totalImagePixels = (long)imageToBinarize->getRawPixelData().size();

	histogram = cudaCalculateHistogram(imageToBinarize->getRawPixelData().data(), totalImagePixels);
	cudaDeviceSynchronize();
	showHistogram(histogram);
	
	unsigned char threshold;
	threshold = cudaFindThreshold(histogram, totalImagePixels);
	cudaDeviceSynchronize();
	printf("\t[GPU] Threshold value: %d", threshold);

	unsigned char* binarizedRawPixels = cudaBinarize(imageToBinarize->getRawPixelData().data(), totalImagePixels, threshold);
	cudaDeviceSynchronize();

	std::vector<unsigned char> binarizedVector(binarizedRawPixels[0], binarizedRawPixels[totalImagePixels]);

	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		binarizedVector
	);
}

void OtsuBinarizer::showHistogram(unsigned int* histogram) {
	printf("Histogram:\n");
	unsigned int value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %d\n", i, value);
	}
}
