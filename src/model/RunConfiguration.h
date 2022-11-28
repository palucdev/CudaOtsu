#pragma once
#include <string>
#include "PngImage.h"

class RunConfiguration
{
private:
	std::string fullFilePath;
	int threadsPerBlock;
	int numBlocks;
	int cpuThreads;
	bool drawHistograms;
	PngImage* loadedImage;
	RunConfiguration();
public:
	~RunConfiguration();
	friend class RunConfigurationBuilder;

	std::string getFullFilePath();
	int getThreadsPerBlock();
	int getNumberOfBlocks();
	int getCpuThreads();
	bool shouldDrawHistograms();
	PngImage* getLoadedImage();
	bool hasLoadedImage();
	void print();
};