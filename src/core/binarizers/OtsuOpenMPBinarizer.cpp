#include "../../utils/ImageFileUtil.h"
#include "OtsuOpenMPBinarizer.h"
#include <omp.h>
#include <stdio.h>
#include <time.h>

OtsuOpenMPBinarizer::OtsuOpenMPBinarizer(PngImage* imageToBinarize)
{
	this->imageToBinarize = imageToBinarize;
}

MethodImplementation OtsuOpenMPBinarizer::getBinarizerType()
{
	return CPU_OpenMP;
}

const char* OtsuOpenMPBinarizer::getBinarizedFilePrefix()
{
	return "cpu-openmp_binarized_";
}

BinarizationResult* OtsuOpenMPBinarizer::binarize(RunConfiguration* runConfig)
{
	printf("\nSetting OpenMP threads num to %d threads\n", runConfig->getCpuThreads());
	omp_set_dynamic(0);
	omp_set_num_threads(runConfig->getCpuThreads());

	this->histogram = calculateHistogram();
	this->cpuThreads = runConfig->getCpuThreads();

	ExecutionTimestamp* executionTimestamp = new ExecutionTimestamp();
	clock_t time;
	time = clock();

	std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(runConfig->getFullFilePath(), getBinarizedFilePrefix());

	PngImage *cpuBinarizedImage = binarize();

	time = clock() - time;

	executionTimestamp->binarizationTimeInSeconds = ((double)time / CLOCKS_PER_SEC);

	printf("\nCPU-OpenMP binarization taken %f seconds\n", ((double)time / CLOCKS_PER_SEC));

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

	delete cpuBinarizedImage;

	return new BinarizationResult(
		getBinarizerType(),
		cpuBinarizedFilename.c_str(),
		executionTimestamp
	);
}

std::vector<double> OtsuOpenMPBinarizer::calculateHistogram() {
	std::vector<double> histogram(PngImage::MAX_PIXEL_VALUE);
	std::vector<unsigned char> image = this->imageToBinarize->getRawPixelData();

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

	return histogram;
}

int OtsuOpenMPBinarizer::findThreshold() {
	int threshold;
	long int totalPixels = this->imageToBinarize->getTotalPixels();
	double* betweenClassVariances = new double[PngImage::MAX_PIXEL_VALUE];
	double allProbabilitySum = 0;
	std::vector<double> localHistogram = this->histogram;

	#pragma omp parallel shared(allProbabilitySum, betweenClassVariances, totalPixels, localHistogram) num_threads(cpuThreads)
	{
		double firstClassProbability = 0, secondClassProbability = 0;
		double firstClassMean = 0, secondClassMean = 0, firstProbabilitySum = 0;

		int chunkSize = PngImage::MAX_PIXEL_VALUE / omp_get_num_threads();

		#pragma omp for schedule(static, chunkSize)
		for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
			#pragma omp atomic
			allProbabilitySum += i * localHistogram[i];
			betweenClassVariances[i] = 0;
		}

		#pragma omp barrier

		#pragma omp for schedule(static, chunkSize)
		for (int v = 0; v < PngImage::MAX_PIXEL_VALUE; v++) {
			firstClassProbability = 0;
			firstProbabilitySum = 0;
			for (int t = 0; t <= v % PngImage::MAX_PIXEL_VALUE; t++) {
				firstClassProbability += localHistogram[t];
				firstProbabilitySum += t * localHistogram[t];
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

	delete[] betweenClassVariances;

	return threshold;
}

PngImage* OtsuOpenMPBinarizer::binarize() {
	int threshold = findThreshold();
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

void OtsuOpenMPBinarizer::showHistogram() {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < PngImage::MAX_PIXEL_VALUE; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}
