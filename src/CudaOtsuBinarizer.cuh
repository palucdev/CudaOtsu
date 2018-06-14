#include "model/PngImage.h"
#include "model/ExecutionTimestamp.h"

#pragma once
class CudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	std::string getBinarizerExecutionInfo(std::string fileName);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram, const char* TAG = "GPU");
	virtual ~CudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	int numBlocks_;
	ExecutionTimestamp* binarizerTimestamp_;
	bool drawHistogram_;
	const char* TAG;
	virtual void showHistogram(double* histogram);
	virtual double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	virtual unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	virtual unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

