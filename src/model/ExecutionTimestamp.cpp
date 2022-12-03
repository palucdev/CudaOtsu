#include "ExecutionTimestamp.h"

#include "../utils/ImageFileUtil.h"
#include <vector>

ExecutionTimestamp::ExecutionTimestamp()
{
	this->histogramBuildingTimeInSeconds = 0;
	this->thresholdFindingTimeInSeconds = 0;
	this->binarizationTimeInSeconds = 0;
}


ExecutionTimestamp::~ExecutionTimestamp()
{
}

// Format: fileName,TAG,histogramBuildingTime,thresholdFindingTime,binarizationTime,executionTime
std::string ExecutionTimestamp::toCommaSeparatedRow(std::string fileName, std::string tag)
{
	std::vector<std::string> values;
	values.push_back(fileName);
	values.push_back(tag);
	values.push_back(std::to_string(histogramBuildingTimeInSeconds));
	values.push_back(std::to_string(thresholdFindingTimeInSeconds));
	values.push_back(std::to_string(binarizationTimeInSeconds));
	values.push_back(std::to_string(getExecutionTime()));

	return ImageFileUtil::joinString(values, ',');
}

double ExecutionTimestamp::getExecutionTime()
{
	return histogramBuildingTimeInSeconds + thresholdFindingTimeInSeconds + binarizationTimeInSeconds;
}
