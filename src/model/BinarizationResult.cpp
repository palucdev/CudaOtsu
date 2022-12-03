#include "BinarizationResult.h"

BinarizationResult::BinarizationResult(unsigned int method, std::string binarizedImagePath, ExecutionTimestamp* executionTimestamp)
{
    this->method = method;
    this->binarizedImagePath = binarizedImagePath;
    this->executionTimestamp = executionTimestamp;
}


BinarizationResult::~BinarizationResult() {};

void BinarizationResult::printResult()
{
    printf("\n---------------------------------------------\n");
    printf("\nBinarization result:\n");
	printf("\tMethod: %d\n", this->method);
	printf("\tBinarized image path: %s\n", this->binarizedImagePath.c_str());

    if (this->executionTimestamp->histogramBuildingTimeInSeconds != 0) {
	    printf("\tExecution - histogram build time: %f seconds\n", this->executionTimestamp->histogramBuildingTimeInSeconds);
    }

    if (this->executionTimestamp->thresholdFindingTimeInSeconds != 0) {
	    printf("\tExecution - threshold lookup time: %f seconds\n", this->executionTimestamp->thresholdFindingTimeInSeconds);
    }

    if (this->executionTimestamp->binarizationTimeInSeconds != 0) {
	    printf("\tExecution - binarization time: %f seconds\n", this->executionTimestamp->binarizationTimeInSeconds);
    }
    printf("\n---------------------------------------------\n");
}