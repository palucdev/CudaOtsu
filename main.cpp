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
	const char* binarizedFilename = "assets/binarized_copy.png";

	PngImage* loadedImage = ImageFileUtil::loadPngFile(filename);

	PngImage* binarizedImage = OtsuBinarizer::binarizeOnCpu(loadedImage);

	ImageFileUtil::savePngFile(binarizedImage, binarizedFilename);

	// to test if CUDA kernels work
	OtsuBinarizer::binarizeOnGpu(loadedImage);
	
	system("pause");
	return 0;
}