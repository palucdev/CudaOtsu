// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

// Memory leaks checking
// #include "vld.h"

#include "libs/lodepng.h"
#include "utils/ImageFileUtil.h"
#include "utils/CudaUtil.h"
#include "model/PngImage.h"
#include "OtsuBinarizer.h"
#include "CudaOtsuBinarizer.cuh"
#include "SMCudaOtsuBinarizer.cuh"
#include "MonoCudaOtsuBinarizer.cuh"

void printHelp() {
	std::string helpMessage = "";
	helpMessage.append("Help:\n");
	helpMessage.append("<program> filePath cudaThreadsNumber cudaBlocksNumber [optional flags]\n");
	helpMessage.append("\tFlags:\n");
	helpMessage.append("\t\t -h show histogram values for each binarizer run\n");
	helpMessage.append("\t\t -d <deviceName> choose GPU device by given name (defaults to 0)\n");

	printf(helpMessage.c_str());

}

// CudaOtsu filepath/dirpath threadsPerBlock numBlocks
int main(int argc, char **argv)
{
	static const int DEFAULT_THREADS_NUMBER = 512;
	static const int DEFAULT_BLOCKS_NUMBER = 512;

	std::string fullFilePath;
	int threadsPerBlock;
	int numBlocks;
	bool drawHistograms = false;
	int cudaDeviceId;

	if (argc <= 3) {
		printHelp();
		CudaUtil::getAvailableGpuNames();
		return -1;
	}
	else {
		fullFilePath = argv[1];
		threadsPerBlock = std::atoi(argv[2]) > 0 ? std::atoi(argv[2]) : DEFAULT_THREADS_NUMBER;
		numBlocks = std::atoi(argv[3]) > 0 ? std::atoi(argv[3]) : DEFAULT_BLOCKS_NUMBER;

		for (int argumentIndex = 4; argumentIndex < argc; argumentIndex++) {
			std::string flag(argv[argumentIndex]);

			if (flag == "-h")
				drawHistograms = true;

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
				}
			}
		}
	}
	
	PngImage* loadedImage = ImageFileUtil::loadPngFile(fullFilePath.c_str());

	if (loadedImage != nullptr) {
		
		std::string cpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "cpu_binarized_");

		PngImage* cpuBinarizedImage = OtsuBinarizer::binarize(loadedImage);

		ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename.c_str());

		delete cpuBinarizedImage; 
	
		CudaOtsuBinarizer* cudaBinarizer = new CudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistograms);

		PngImage* gpuBinarizedImage = cudaBinarizer->binarize(loadedImage);

		std::string gpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_binarized_");

		ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename.c_str());

		delete gpuBinarizedImage; 
		delete cudaBinarizer;

		SMCudaOtsuBinarizer* smCudaBinarizer = new SMCudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistograms);

		PngImage* sharedMemoryGpuBinarizedImage = smCudaBinarizer->binarize(loadedImage);

		std::string smGpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_shared_memory_binarized_");

		ImageFileUtil::savePngFile(sharedMemoryGpuBinarizedImage, smGpuBinarizedFilename.c_str());

		delete sharedMemoryGpuBinarizedImage;
		delete smCudaBinarizer;

		MonoCudaOtsuBinarizer* monoCudaBinarizer = new MonoCudaOtsuBinarizer(threadsPerBlock, drawHistograms);

		PngImage* monoKernelGpuBinarizedImage = monoCudaBinarizer->binarize(loadedImage);

		std::string monoKernelGpuBinarizedFilename = ImageFileUtil::addPrefix(fullFilePath, "gpu_mono_binarized_");

		ImageFileUtil::savePngFile(monoKernelGpuBinarizedImage, monoKernelGpuBinarizedFilename.c_str());

		delete monoKernelGpuBinarizedImage;
		delete monoCudaBinarizer;
	}

	delete loadedImage;

	return 0;
}
