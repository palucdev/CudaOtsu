#include "../../model/PngImage.h"
#include "OtsuBinarizer.h"

#pragma once
class OtsuOpenMPBinarizer: public AbstractBinarizer
{
public:
	BinarizationResult *binarize(RunConfiguration *runConfig);
	OtsuOpenMPBinarizer(PngImage* imageToBinarize);

private:
	PngImage* imageToBinarize;
	std::vector<double> histogram;
	int cpuThreads;

	MethodImplementation getBinarizerType();
	const char* getBinarizedFilePrefix();
	std::vector<double> calculateHistogram();
	int findThreshold();
	PngImage* binarize();
	void showHistogram();
};