#include <string>
#include "ExecutionTimestamp.h"

#pragma once
class BinarizationResult
{
private:
    unsigned int method;
    std::string binarizedImagePath;
    ExecutionTimestamp *executionTimestamp;

public:
    BinarizationResult(unsigned int method, std::string binarizedImagePath, ExecutionTimestamp* executionTimestamp);
    ~BinarizationResult();
};
