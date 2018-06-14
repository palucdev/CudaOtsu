#include "ExecutionTimestamp.h"

#include "../utils/ImageFileUtil.h"
#include <vector>

ExecutionTimestamp::ExecutionTimestamp()
{
	this->histogramBuildingTime = 0;
	this->thresholdFindingTime = 0;
	this->binarizationTime = 0;
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
	values.push_back(std::to_string(histogramBuildingTime));
	values.push_back(std::to_string(thresholdFindingTime));
	values.push_back(std::to_string(binarizationTime));
	values.push_back(std::to_string(getExecutionTime()));

	return ImageFileUtil::joinString(values, ',');
}

double ExecutionTimestamp::getExecutionTime()
{
	return histogramBuildingTime + thresholdFindingTime + binarizationTime;
}
