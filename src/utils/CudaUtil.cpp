#include "CudaUtil.h"
#include <string.h>

// CUDA imports
#include <cuda_runtime.h>

bool CudaUtil::isGpuAvailable()
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for (int deviceIndex = 0; deviceIndex < devicesCount; deviceIndex++) {
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);

		if (deviceProperties.major >= 2
			&& deviceProperties.minor >= 0)
		{
			return true;
		}
	}

	return false;
}

void CudaUtil::getAvailableGpuNames()
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	printf("\nAvailable GPUs:\n");
	for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		printf("[%d] %s\n", deviceIndex, deviceProperties.name);
	}
}

bool CudaUtil::setGpu(int deviceIndex)
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	if (deviceIndex < devicesCount) {
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		printf("Selected GPU: (%d) - %s\n", deviceIndex, deviceProperties.name);
		printf("Compute Capability: %d.%d\n", deviceProperties.major, deviceProperties.minor);
		cudaSetDevice(deviceIndex);
		return true;
	} else {
		printf("No GPU available for given index: %d\n", deviceIndex);
		return false;
	}
}

CudaUtil::CudaUtil() {}

int CudaUtil::getDeviceIndexForName(std::string deviceName)
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		if (deviceProperties.name == deviceName)
		{
			return deviceIndex;
		}
	}

	return -1;
}
