#include "RunConfigurationBuilder.h"

RunConfigurationBuilder &RunConfigurationBuilder::forFileInPath(std::string fullFilePath)
{
	runConfiguration.fullFilePath = fullFilePath;
	return *this;
}

RunConfigurationBuilder &RunConfigurationBuilder::withThreadsPerBlock(int threadsPerBlock)
{
	runConfiguration.threadsPerBlock = threadsPerBlock;
	return *this;
}

RunConfigurationBuilder &RunConfigurationBuilder::withNumberOfBlocks(int numBlocks)
{
	runConfiguration.numBlocks = numBlocks;
	return *this;
}

RunConfigurationBuilder &RunConfigurationBuilder::withCpuThreads(int cpuThreads)
{
	runConfiguration.cpuThreads = cpuThreads;
	return *this;
}

RunConfigurationBuilder &RunConfigurationBuilder::withHistograms(bool drawHistograms)
{
	runConfiguration.drawHistograms = drawHistograms;
	return *this;
}

RunConfigurationBuilder &RunConfigurationBuilder::forImage(PngImage *loadedImage)
{
	runConfiguration.loadedImage = loadedImage;
	return *this;
}

RunConfiguration RunConfigurationBuilder::build()
{
	return this->runConfiguration;
}