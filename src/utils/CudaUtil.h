#include <string>

#pragma once
class CudaUtil
{
public:
	static bool isGpuAvailable();
	static bool setGpu(int deviceId);
	static void getAvailableGpuNames();
	static std::string getDeviceName(int deviceId);
private: 
	CudaUtil();
	static int getDeviceIndexForName(std::string deviceName);
};

