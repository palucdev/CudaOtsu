#include "model/PngImage.h" 
#include "CudaOtsuBinarizer.cuh"

#pragma once
class SMCudaOtsuBinarizer : public CudaOtsuBinarizer
{
public:
	SMCudaOtsuBinarizer(int threadsPerBlock, int numBlocks);
	virtual ~SMCudaOtsuBinarizer();
protected:
	unsigned char cudaFindThreshold(double* histogram, long int totalPixels) override;
	double* SMCudaOtsuBinarizer::cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels) override;
};
