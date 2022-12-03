#include "../../model/PngImage.h"
#include "../../interface/AbstractBinarizer.h" 

#pragma once
class OtsuBinarizer : public AbstractBinarizer
{
public:
	BinarizationResult *binarize(RunConfiguration *runConfig);
	OtsuBinarizer(PngImage* imageToBinarize);
	
private:
	PngImage* imageToBinarize;
	std::vector<double> histogram;

	MethodImplementation getBinarizerType();
	const char* getBinarizedFilePrefix();
	std::vector<double> calculateHistogram();
	int findThreshold();
	PngImage* binarize();
	void showHistogram();
};

