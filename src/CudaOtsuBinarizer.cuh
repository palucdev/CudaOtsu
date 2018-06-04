#include "model/PngImage.h" 

#pragma once
class CudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks);
	~CudaOtsuBinarizer();
private:
	int threadsPerBlock_;
	int numBlocks_;
	void showHistogram(double* histogram);
	double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

