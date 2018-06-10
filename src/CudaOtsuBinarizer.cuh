#include "model/PngImage.h" 

#pragma once
class CudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram, const char* TAG = "GPU");
	~CudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	int numBlocks_;
	float executionTime_;
	bool drawHistogram_;
	const char* TAG;
	virtual void showHistogram(double* histogram);
	virtual double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	virtual unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	virtual unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

