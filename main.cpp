// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "lodepng.h"
#include "ImageFileUtil.h"
#include "PngImage.h"
#include "OtsuBinarizer.h"

int main(int argc, char **argv)
{
	const char* filename = "assets/example_grey.png";
	const char* cpuBinarizedFilename = "assets/cpu_binarized_copy.png";
	const char* gpuBinarizedFilename = "assets/gpu_binarized_copy.png";

	PngImage* loadedImage = ImageFileUtil::loadPngFile(filename);

	PngImage* cpuBinarizedImage = OtsuBinarizer::binarizeOnCpu(loadedImage);

	ImageFileUtil::savePngFile(cpuBinarizedImage, cpuBinarizedFilename);

	PngImage* gpuBinarizedImage = OtsuBinarizer::binarizeOnGpu(loadedImage);
	
	ImageFileUtil::savePngFile(gpuBinarizedImage, gpuBinarizedFilename);

	system("pause");
	return 0;
}