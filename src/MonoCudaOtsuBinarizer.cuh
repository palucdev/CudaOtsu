#include "model/PngImage.h" 

#pragma once
class MonoCudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	MonoCudaOtsuBinarizer(int threadsPerBlock, bool drawHistogram, const char* TAG = "GPU - Single Kernel");
	~MonoCudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	float executionTime_;
	bool showHistogram_;
	const char* TAG;
	void showHistogram(double* histogram);
	unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels);
};

