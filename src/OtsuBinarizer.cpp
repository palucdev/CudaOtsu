#include "OtsuBinarizer.h"
#include <stdio.h>

OtsuBinarizer::OtsuBinarizer(){}

PngImage* OtsuBinarizer::binarize(PngImage * imageToBinarize)
{
	std::vector<double> histogram(PngImage::MAX_PIXEL_VALUE);

	calculateHistogram(imageToBinarize->getRawPixelData(), histogram);

	showHistogram(histogram.data());

	int threshold = findThreshold(histogram, imageToBinarize->getTotalPixels());

	printf("\t[CPU] Threshold value: %d", threshold);

	return binarizeImage(imageToBinarize, threshold);
}

void OtsuBinarizer::calculateHistogram(const std::vector<unsigned char>& image, std::vector<double>& histogram) {
	std::vector<unsigned char> occurences(PngImage::MAX_PIXEL_VALUE);
	unsigned char pixelValue;
	long totalPixels = image.size();

	for (std::vector<unsigned char>::size_type i = 0; i != totalPixels; i++) {
		pixelValue = image[i];
		histogram[pixelValue]++;
	}

	// Normalization
	for (std::vector<unsigned char>::size_type v = 0; v != PngImage::MAX_PIXEL_VALUE; v++) {
		histogram[v] /= totalPixels;
	}
}

int OtsuBinarizer::findThreshold(std::vector<double>& histogram, long int totalPixels) {
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
		secondClassProbability = 1 - firstClassProbability;

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

void OtsuBinarizer::showHistogram(double* histogram) {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}
