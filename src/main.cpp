// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <time.h>
#include <omp.h>

// Memory leaks checking
// #include "vld.h"

#include "libs/lodepng.h"
#include "utils/ImageFileUtil.h"
#include "utils/CudaUtil.h"
#include "model/PngImage.h"
#include "OtsuBinarizer.h"
#include "OtsuOpenMPBinarizer.h"
#include "CudaOtsuBinarizer.cuh"
#include "SMCudaOtsuBinarizer.cuh"
#include "MonoCudaOtsuBinarizer.cuh"

enum MethodImplementations : unsigned int {
	CPU,
	CPU_OpenMP,
	GPU,
	GPU_SharedMemory,
	GPU_MonoKernel,
	ALL
};


std::string getConfigurationInfo(int threadsPerBlock, int numBlocks) {
	std::vector<std::string> params;
	params.push_back(std::to_string(threadsPerBlock));
	params.push_back(std::to_string(numBlocks));
	params.push_back(CudaUtil::getDeviceName(CudaUtil::getCurrentDevice()));

	return ImageFileUtil::joinString(params, ',');
}

void runCpuImplementation(std::string fullFilePath, PngImage* loadedImage) {

	clock_t time;
	time = clock();

	std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "cpu_binarized_");

	PngImage* cpuBinarizedImage = OtsuBinarizer::binarize(loadedImage);

	time = clock() - time;

	printf("\nCPU binarization took %f seconds\n", ((double)time / CLOCKS_PER_SEC));

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

	delete cpuBinarizedImage;
}

void runCpuOpenmpImplementation(std::string fullFilePath, PngImage* loadedImage, int cpuThreads) {

	printf("\nSetting OpenMP threads num to %d threads\n", cpuThreads);
	omp_set_dynamic(0);
	omp_set_num_threads(cpuThreads);

	clock_t time;
	time = clock();

	std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "cpu-openmp_binarized_");

	PngImage* cpuBinarizedImage = OtsuOpenMPBinarizer::binarize(loadedImage, cpuThreads);

	time = clock() - time;

	printf("\nCPU-OpenMP binarization taken %f seconds\n", ((double)time / CLOCKS_PER_SEC));

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

	delete cpuBinarizedImage;
}

std::string runGpuImplementation(std::string fullFilePath, PngImage* loadedImage, int threadsPerBlock, int numBlocks, bool drawHistograms) {

	CudaOtsuBinarizer* cudaBinarizer = new CudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistograms);

	PngImage* gpuBinarizedImage = cudaBinarizer->binarize(loadedImage);

	std::string gpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_binarized_");

	ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename.c_str());

	std::string csvTimesLog = cudaBinarizer->getBinarizerExecutionInfo(fullFilePath);
	std::string configLog = getConfigurationInfo(threadsPerBlock, numBlocks);

	delete gpuBinarizedImage;
	delete cudaBinarizer;

	return csvTimesLog + "," + configLog;
}

std::string runGpuSharedMemoryImplementation(std::string fullFilePath, PngImage* loadedImage, int threadsPerBlock, int numBlocks, bool drawHistograms) {

	SMCudaOtsuBinarizer* smCudaBinarizer = new SMCudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistograms);

	PngImage* sharedMemoryGpuBinarizedImage = smCudaBinarizer->binarize(loadedImage);

	std::string smGpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_shared_memory_binarized_");

	ImageFileUtil::savePngFile(sharedMemoryGpuBinarizedImage, smGpuBinarizedFilename.c_str());

	std::string csvTimesLog = smCudaBinarizer->getBinarizerExecutionInfo(fullFilePath);
	std::string configLog = getConfigurationInfo(threadsPerBlock, numBlocks);
	
	delete sharedMemoryGpuBinarizedImage;
	delete smCudaBinarizer;

	return csvTimesLog + "," + configLog;
}

void runGpuMonoKernelImplementation(std::string fullFilePath, PngImage* loadedImage, int threadsPerBlock, int numBlocks, bool drawHistograms) {

	MonoCudaOtsuBinarizer* monoCudaBinarizer = new MonoCudaOtsuBinarizer(threadsPerBlock, drawHistograms);

	PngImage* monoKernelGpuBinarizedImage = monoCudaBinarizer->binarize(loadedImage);

	std::string monoKernelGpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_mono_binarized_");

	ImageFileUtil::savePngFile(monoKernelGpuBinarizedImage, monoKernelGpuBinarizedFilename.c_str());

	delete monoKernelGpuBinarizedImage;
	delete monoCudaBinarizer;
}

void printHelp() {
	std::string helpMessage = "";
	helpMessage.append("Help:\n");
	helpMessage.append("<program> filePath cudaThreadsNumber cudaBlocksNumber [optional flags]\n");
	helpMessage.append("\tFlags:\n");
	helpMessage.append("\t\t -h show histogram values for each binarizer run\n");
	helpMessage.append("\t\t -d <deviceName> choose GPU device by given name (defaults to 0)\n");
	helpMessage.append("\t\t --cpu run CPU version of algorithm\n");
	helpMessage.append("\t\t --cpu-openmp run CPU with OpenMP version of algorithm\n");
	helpMessage.append("\t\t --gpu run GPU reference version of algorithm\n");
	helpMessage.append("\t\t --gpu-sm run GPU version of algorithm with shared memory optimization\n");
	helpMessage.append("\t\t --gpu-mono run GPU version of algorithm with single kernel arch on single block\n");
	helpMessage.append("\t\t --run-all run all implemented versions of Otsu algorithm (CPU and GPU)\n");

	printf(helpMessage.c_str());

}

// CudaOtsu filepath/dirpath threadsPerBlock numBlocks
int main(int argc, char **argv)
{
	static const int DEFAULT_THREADS_NUMBER = 512;
	static const int DEFAULT_BLOCKS_NUMBER = 512;
	static const int DEFAULT_CPU_THREADS = 16;

	std::string fullFilePath;
	int threadsPerBlock, numBlocks, cpuThreads;
	bool drawHistograms = false;
	int cudaDeviceId;

	std::vector<std::string> binarizerTimestamps;
	const char* timestampsFile = "times.csv";

	// Default CudaOtsuBinarizer usage
	bool algChosenToRun[6] = { false, true, false, false, false, false };

	if (argc <= 3) {
		printHelp();
		CudaUtil::getAvailableGpuNames();
		return -1;
	}
	else {
		fullFilePath = argv[1];
		threadsPerBlock = std::atoi(argv[2]) > 0 ? std::atoi(argv[2]) : DEFAULT_THREADS_NUMBER;
		numBlocks = std::atoi(argv[3]) > 0 ? std::atoi(argv[3]) : DEFAULT_BLOCKS_NUMBER;
		cpuThreads = std::atoi(argv[4]) > 0 ? std::atoi(argv[4]) : DEFAULT_CPU_THREADS;

		for (int argumentIndex = 5; argumentIndex < argc; argumentIndex++) {
			std::string flag(argv[argumentIndex]);

			if (flag == "-h") {
				drawHistograms = true;
				continue;
			}

			if (flag == "-d") {
				int nextArgument = argumentIndex + 1;
				if (nextArgument < argc) {
					cudaDeviceId = std::atoi(argv[nextArgument]);

					bool gpuSetSuccess = CudaUtil::setGpu(cudaDeviceId);

					if (!gpuSetSuccess) {
						CudaUtil::getAvailableGpuNames();
						return -1;
					}

					argumentIndex = nextArgument;
					continue;
				}
			}

			if (flag == "--cpu") {
				algChosenToRun[CPU] = true;
				continue;
			}

			if (flag == "--cpu-openmp") {
				algChosenToRun[CPU_OpenMP] = true;
				continue;
			}
				
			if (flag == "--gpu") {
				algChosenToRun[GPU] = true;
				continue;
			}
				
			if (flag == "--gpu-sm") {
				algChosenToRun[GPU_SharedMemory] = true;
				continue;
			}
				
			if (flag == "--gpu-mono") {
				algChosenToRun[GPU_MonoKernel] = true;
				continue;
			}
	
			if (flag == "--run-all") {
				algChosenToRun[ALL] = true;
				continue;
			}
		}
	}
	
	PngImage* loadedImage = ImageFileUtil::loadPngFile(fullFilePath.c_str());

	if (loadedImage != nullptr) {

		if (algChosenToRun[CPU] || algChosenToRun[ALL]) {
			runCpuImplementation(fullFilePath, loadedImage);
		}

		if (algChosenToRun[CPU_OpenMP] || algChosenToRun[ALL]) {
			runCpuOpenmpImplementation(fullFilePath, loadedImage, cpuThreads);
		}

		if (algChosenToRun[GPU] || algChosenToRun[ALL]) {
			std::string csvTimeLog = runGpuImplementation(fullFilePath, loadedImage, threadsPerBlock, numBlocks, drawHistograms);
			binarizerTimestamps.push_back(csvTimeLog);
		}

		if (algChosenToRun[GPU_SharedMemory] || algChosenToRun[ALL]) {
			std::string csvTimeLog = runGpuSharedMemoryImplementation(fullFilePath, loadedImage, threadsPerBlock, numBlocks, drawHistograms);
			binarizerTimestamps.push_back(csvTimeLog);
		}

		if (algChosenToRun[GPU_MonoKernel] || algChosenToRun[ALL]) {
			runGpuMonoKernelImplementation(fullFilePath, loadedImage, threadsPerBlock, numBlocks, drawHistograms);
		}
	}

	delete loadedImage;

	ImageFileUtil::saveCsvFile(binarizerTimestamps, timestampsFile);

	return 0;
}
