#include <string>

#pragma once
class ExecutionTimestamp
{
public:
	ExecutionTimestamp();
	~ExecutionTimestamp();
	double histogramBuildingTimeInSeconds;
	double thresholdFindingTimeInSeconds;
	double binarizationTimeInSeconds;
	std::string toCommaSeparatedRow(std::string fileName, std::string tag);
	double getExecutionTime();
};

