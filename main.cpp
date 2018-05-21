// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "lodepng.h"
#include "ImageFileUtil.h"
#include "PngImage.h"

// CUDA Runtime
#include <cuda_runtime.h>

#define PIXEL_VALUE_RANGE 256

extern "C" void cudaCalculateHistogram();
extern "C" void cudaBinarize();

void calculateHistogram(std::vector<unsigned char>& image, std::vector<double>& histogram) {
	std::vector<unsigned char> occurences(PIXEL_VALUE_RANGE);
	unsigned char pixelValue;

	for (std::vector<unsigned char>::size_type i = 0; i != image.size(); i++) {
		pixelValue = image[i];
		histogram[pixelValue]++;
	}
}

int findThreshold(std::vector<double>& histogram, long int totalPixels) {
	int threshold;
	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double betweenClassVariance = 0, maxVariance = 0;
	double allProbabilitySum = 0, firstProbabilitySum = 0;

	for (int i = 0; i < PIXEL_VALUE_RANGE; i++) {
		allProbabilitySum += i * histogram[i];
	}

	for (int t = 0; t < PIXEL_VALUE_RANGE; t++) {
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

PngImage* binarizeImage(PngImage* imageToBinarize, int threshold) {
	std::vector<unsigned char> imagePixels = imageToBinarize->getRawPixelData();
	for (std::vector<unsigned char>::size_type i = 0; i != imageToBinarize->getTotalPixels(); i++) {
		if ((int)imagePixels[i] > threshold) {
			imagePixels[i] = (unsigned char)255;
		} else {
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

int main(int argc, char **argv)
{
	//Kernel configuration, where a two-dimensional grid and
	//three-dimensional blocks are configured.

	//std::vector<unsigned char> image; // raw pixels
	std::vector<double> histogram(PIXEL_VALUE_RANGE);

	const char* filename = "assets/example_grey.png";
	const char* binarizedFilename = "assets/binarized_copy.png";
	unsigned width = 0;
	unsigned height = 0;

	PngImage* loadedImage = ImageFileUtil::loadPngFile(filename);

	calculateHistogram(loadedImage->getRawPixelData(), histogram);

	int threshold = findThreshold(histogram, loadedImage->getTotalPixels());

	PngImage* binarizedImage = binarizeImage(loadedImage, threshold);

	ImageFileUtil::savePngFile(binarizedImage, binarizedFilename);
	
	cudaCalculateHistogram();
	cudaDeviceSynchronize();
	cudaBinarize();
	cudaDeviceSynchronize();

	system("pause");

	return 0;
}