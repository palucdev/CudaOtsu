#include <vector>

#pragma once
class ImageFileUtil
{
private:
	ImageFileUtil();
public:
	static void loadPngFile(const char* filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height);
	static void savePngFile(const char* filename, std::vector<unsigned char>& image, unsigned* width, unsigned* height);
};

