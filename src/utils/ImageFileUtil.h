#include <vector>
#include <string>
#include "../model/PngImage.h"

#pragma once
class ImageFileUtil
{
public:
	static PngImage* loadPngFile(const char* filename);
	static void savePngFile(PngImage* pngImage, const char* newFileName);
	static std::string addPrefix(std::string fullFilePath, const char* prefix);
	static std::string joinString(std::vector<std::string> strings, const char delimiter = '\0');
	static void saveCsvFile(std::vector<std::string> rows, const char* filename);
private:
	ImageFileUtil();
	static std::vector<std::string> splitString(std::string stringToSplit, const char delimiter);
	static bool fileExists(const char * fileName);
	static char getOsPathDelimiter();
};

