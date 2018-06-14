#include <string>

#pragma once
class ExecutionTimestamp
{
public:
	ExecutionTimestamp();
	~ExecutionTimestamp();
	double histogramBuildingTime;
	double thresholdFindingTime;
	double binarizationTime;
	std::string toCommaSeparatedRow(std::string fileName, std::string tag);
	double getExecutionTime();
};

