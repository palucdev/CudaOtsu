#include "../../utils/ImageFileUtil.h"
#include "OtsuBinarizer.h"
#include <stdio.h>
#include <time.h>

OtsuBinarizer::OtsuBinarizer(PngImage* imageToBinarize)
{
	this->imageToBinarize = imageToBinarize;
	this->histogram = calculateHistogram();
}

MethodImplementation OtsuBinarizer::getBinarizerType()
{
	return CPU;
}

BinarizationResult* OtsuBinarizer::binarize(RunConfiguration* runConfig)
{
	ExecutionTimestamp* executionTimestamp = new ExecutionTimestamp();
	clock_t time;
	time = clock();

	std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), "cpu_binarized_");

	PngImage *cpuBinarizedImage = binarize();

	time = clock() - time;

	executionTimestamp->binarizationTimeInSeconds = ((double)time / CLOCKS_PER_SEC);

	printf("\nCPU binarization took %f seconds\n", ((double)time / CLOCKS_PER_SEC));

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

	delete cpuBinarizedImage;

	return new BinarizationResult(
		getBinarizerType(),
		runConfig->getFullFilePath(),
		executionTimestamp
	);
}

std::vector<double> OtsuBinarizer::calculateHistogram() 
{
	std::vector<double> histogram(PngImage::MAX_PIXEL_VALUE);
	std::vector<unsigned char> image = this->imageToBinarize->getRawPixelData();

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

	return histogram;
}

PngImage* OtsuBinarizer::binarize() {
	int threshold = findThreshold();
	std::vector<unsigned char> imagePixels = this->imageToBinarize->getRawPixelData();
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

int OtsuBinarizer::findThreshold() {
	int threshold;
	long int totalPixels = this->imageToBinarize->getTotalPixels();
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

	printf("\t[CPU] Threshold value: %d", threshold);

	return threshold;
}

void OtsuBinarizer::showHistogram() {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}
