#include "OtsuBinarizer.h"

// CUDA imports
#include <cuda_runtime.h>

extern "C" unsigned int* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
extern "C" void cudaBinarize();

OtsuBinarizer::OtsuBinarizer(){}

PngImage* OtsuBinarizer::binarizeOnCpu(PngImage * imageToBinarize)
{
	std::vector<unsigned int> histogram(OtsuBinarizer::PIXEL_VALUE_RANGE);

	calculateHistogram(imageToBinarize->getRawPixelData(), histogram);

	showHistogram(histogram.data());

	int threshold = findThreshold(histogram, imageToBinarize->getTotalPixels());

	return binarizeImage(imageToBinarize, threshold);
}

void OtsuBinarizer::calculateHistogram(std::vector<unsigned char>& image, std::vector<unsigned int>& histogram) {
	std::vector<unsigned char> occurences(OtsuBinarizer::PIXEL_VALUE_RANGE);
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

	for (int i = 0; i < OtsuBinarizer::PIXEL_VALUE_RANGE; i++) {
		allProbabilitySum += i * histogram[i];
	}

	for (int t = 0; t < OtsuBinarizer::PIXEL_VALUE_RANGE; t++) {
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
			imagePixels[i] = (unsigned char)255;
		}
		else {
			imagePixels[i] = (unsigned char)0;
		}
	}

	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		imagePixels
	);
}

PngImage * OtsuBinarizer::binarizeOnGpu(PngImage * imageToBinarize)
{
	unsigned int* histogram = new unsigned int[256];
	histogram = cudaCalculateHistogram(imageToBinarize->getRawPixelData().data(), imageToBinarize->getRawPixelData().size());
	cudaDeviceSynchronize();
	showHistogram(histogram);
	//cudaBinarize();
	//cudaDeviceSynchronize();
	return nullptr;
}

void OtsuBinarizer::showHistogram(unsigned int* histogram) {
	printf("Histogram:\n");
	unsigned int value = 0;
	for (int i = 0; i < 256; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %d\n", i, value);
	}
}
