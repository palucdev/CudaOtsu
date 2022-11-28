#include <string>
#include "../model/PngImage.h"
#include "../model/RunConfiguration.h"

class RunConfigurationBuilder
{
private:
	RunConfiguration runConfiguration;

public:
	RunConfigurationBuilder& forFileInPath(std::string fullFilePath);
	RunConfigurationBuilder& withThreadsPerBlock(int threadsPerBlock);
	RunConfigurationBuilder& withNumberOfBlocks(int numBlocks);
	RunConfigurationBuilder& withCpuThreads(int cpuThreads);
	RunConfigurationBuilder& withHistograms(bool drawHistograms);
	RunConfigurationBuilder& forImage(PngImage* loadedImage);
	RunConfiguration build();
};