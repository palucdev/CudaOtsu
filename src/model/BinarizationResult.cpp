#include "BinarizationResult.h"

BinarizationResult::BinarizationResult(unsigned int method, std::string binarizedImagePath, ExecutionTimestamp* executionTimestamp)
{
    this->method = method;
    this->binarizedImagePath = binarizedImagePath;
    this->executionTimestamp = executionTimestamp;
}


BinarizationResult::~BinarizationResult() {};