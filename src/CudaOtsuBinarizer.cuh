#include "model/PngImage.h" 

#pragma once
class CudaOtsuBinarizer
{
public:
	static PngImage* binarize(PngImage* imageToBinarize);
private:
	CudaOtsuBinarizer();
	static void showHistogram(double* histogram);
	static double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	static unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	static unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

