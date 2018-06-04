#include "model/PngImage.h" 

#pragma once
class CudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, const char* TAG = "GPU");
	~CudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	int numBlocks_;
	float executionTime_;
	const char* TAG;
	void showHistogram(double* histogram);
	virtual double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	virtual unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	virtual unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

