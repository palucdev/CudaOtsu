#include <vector>
#include "PngImage.h"

#pragma once
class ImageFileUtil
{
public:
	static PngImage* loadPngFile(const char* filename);
	static void savePngFile(PngImage* pngImage, const char* newFileName);

private:
	ImageFileUtil();
};

