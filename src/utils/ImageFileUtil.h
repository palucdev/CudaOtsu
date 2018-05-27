#include <vector>
#include "../model/PngImage.h"

#pragma once
class ImageFileUtil
{
public:
	static PngImage* loadPngFile(const char* filename);
	static void savePngFile(PngImage* pngImage, const char* newFileName);
	static std::string addPrefix(std::string fullFilePath, const char* prefix);

private:
	ImageFileUtil();
	static std::vector<std::string> splitString(std::string stringToSplit, const char delimiter);
	static std::string joinString(std::vector<std::string> strings, const char delimiter = '\0');
	static bool fileExists(const char * fileName);
	static const char ImageFileUtil::getOsPathDelimiter();
};

