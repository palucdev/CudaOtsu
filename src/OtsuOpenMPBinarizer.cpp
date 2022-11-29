#include "OtsuOpenMPBinarizer.h"
#include <omp.h>
#include <stdio.h>

OtsuOpenMPBinarizer::OtsuOpenMPBinarizer() {}

PngImage* OtsuOpenMPBinarizer::binarize(PngImage * imageToBinarize, int cpuThreads)
{
	std::vector<double> histogram(PngImage::MAX_PIXEL_VALUE);

	calculateHistogram(imageToBinarize->getRawPixelData(), histogram, cpuThreads);

	//showHistogram(histogram.data());

	int threshold = findThreshold(histogram, imageToBinarize->getTotalPixels(), cpuThreads);

	printf("\t[CPU-OpenMP] Threshold value: %d\n", threshold);

	return binarizeImage(imageToBinarize, threshold, cpuThreads);
}

void OtsuOpenMPBinarizer::calculateHistogram(const std::vector<unsigned char>& image, std::vector<double>& histogram, int cpuThreads) {
	unsigned char pixelValue;
	long totalPixels = image.size();

	#pragma omp parallel firstprivate(pixelValue) shared(totalPixels, histogram, image) num_threads(cpuThreads)
	{
		int chunkSize = PngImage::MAX_PIXEL_VALUE / omp_get_num_threads();

		#pragma omp for schedule(static, chunkSize)
		for (int i = 0; i < totalPixels; i++) {
			pixelValue = image[i];
			
			#pragma omp atomic
			histogram[pixelValue]++;
		}

		#pragma omp barrier

		// Normalization
		#pragma omp for schedule(static, chunkSize)
		for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
			histogram[v] /= totalPixels;
		}
	}
}

int OtsuOpenMPBinarizer::findThreshold(std::vector<double>& histogram, long int totalPixels, int cpuThreads) {
	int threshold;
	double* betweenClassVariances = new double[PngImage::MAX_PIXEL_VALUE];
	double allProbabilitySum = 0;

	#pragma omp parallel shared(allProbabilitySum, betweenClassVariances, totalPixels, histogram) num_threads(cpuThreads)
	{
		double firstClassProbability = 0, secondClassProbability = 0;
		double firstClassMean = 0, secondClassMean = 0, firstProbabilitySum = 0;

		int chunkSize = PngImage::MAX_PIXEL_VALUE / omp_get_num_threads();

		#pragma omp for schedule(static, chunkSize)
		for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
			#pragma omp atomic
			allProbabilitySum += i * histogram[i];
			betweenClassVariances[i] = 0;
		}

		#pragma omp barrier

		#pragma omp for schedule(static, chunkSize)
		for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
			firstClassProbability = 0;
			firstProbabilitySum = 0;
			for (int t = 0; t <= v % PngImage::MAX_PIXEL_VALUE; t++) {
				firstClassProbability += histogram[t];
				firstProbabilitySum += t * histogram[t];
			}

			secondClassProbability = 1 - firstClassProbability;

			firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
			secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

			betweenClassVariances[v] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
		}

		#pragma omp barrier

		#pragma omp single 
		{
			double maxVariance = 0;
			for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
				if (betweenClassVariances[v] > maxVariance) {
					threshold = v;
					maxVariance = betweenClassVariances[v];
				}
			}
		}
	}

	delete betweenClassVariances;

	return threshold;
}

PngImage* OtsuOpenMPBinarizer::binarizeImage(PngImage* imageToBinarize, int threshold, int cpuThreads) {
	std::vector<unsigned char> imagePixels = imageToBinarize->getRawPixelData();

	int totalPixels = imageToBinarize->getTotalPixels();
	int chunkSize = totalPixels / cpuThreads;

	#pragma omp parallel for shared(totalPixels, imagePixels, threshold) schedule(dynamic, chunkSize) num_threads(cpuThreads)
	for (int i = 0; i < totalPixels; i++) {
		if ((int)imagePixels[i] > threshold) {
			imagePixels[i] = PngImage::COLOR_WHITE;
		}
		else {
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

void OtsuOpenMPBinarizer::showHistogram(double* histogram) {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}
