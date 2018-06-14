#include "model/PngImage.h" 
#include "CudaOtsuBinarizer.cuh"

#pragma once
class SMCudaOtsuBinarizer : public CudaOtsuBinarizer
{
public:
	SMCudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram);
	~SMCudaOtsuBinarizer();
protected:
	unsigned char cudaFindThreshold(double* histogram, long int totalPixels) override;
	double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels) override;
};
