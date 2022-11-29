#pragma once
#include <string>
#include "PngImage.h"
#include "../interface/MethodImplementation.h"

class RunConfiguration
{
private:
	std::string fullFilePath;
	int threadsPerBlock;
	int numBlocks;
	int cpuThreads;
	bool drawHistograms;
	PngImage* loadedImage;

	// Default CudaOtsuBinarizer usage
    bool algChosenToRun[6] = {false, true, false, false, false, false};

public:
	RunConfiguration();
	~RunConfiguration();
	friend class RunConfigurationBuilder;

	std::string getFullFilePath();
	int getThreadsPerBlock();
	int getNumberOfBlocks();
	int getCpuThreads();
	bool shouldDrawHistograms();
	PngImage* getLoadedImage();
	bool hasLoadedImage();
	bool shouldRunAlgorithm(unsigned int algorithm);
	void print();
};