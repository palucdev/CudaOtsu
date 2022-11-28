#include "RunConfiguration.h"

RunConfiguration::RunConfiguration() {}

RunConfiguration::~RunConfiguration()
{
	delete this->loadedImage;
}

std::string RunConfiguration::getFullFilePath()
{
	return this->fullFilePath;
}

int RunConfiguration::getThreadsPerBlock()
{
	return this->threadsPerBlock;
}

int RunConfiguration::getNumberOfBlocks()
{
	return this->numBlocks;
}

int RunConfiguration::getCpuThreads()
{
	return this->cpuThreads;
}

bool RunConfiguration::shouldDrawHistograms()
{
	return this->drawHistograms;
}

PngImage *RunConfiguration::getLoadedImage()
{
	return this->loadedImage;
}

bool RunConfiguration::hasLoadedImage()
{
	return this->loadedImage != nullptr;
}

void RunConfiguration::print()
{
	printf("\nRun configuration:\n");
	printf("Full file path: %s\n", this->getFullFilePath().c_str());
	printf("Number of blocks: %d\n", this->getNumberOfBlocks());
	printf("Threads per block: %d\n", this->getThreadsPerBlock());
	printf("CPU threads: %d\n", this->getCpuThreads());
	printf("Should draw histograms: %d\n", this->shouldDrawHistograms());
	printf("Image loaded: %d\n", this->hasLoadedImage());
}