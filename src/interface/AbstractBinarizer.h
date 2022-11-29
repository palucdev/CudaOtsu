#include "../model/BinarizationResult.h"
#include "../model/RunConfiguration.h"

class AbstractBinarizer
{
public:
    virtual BinarizationResult *binarize(RunConfiguration *runConfig);
};