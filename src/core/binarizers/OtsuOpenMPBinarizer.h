#include "../../model/PngImage.h" 

#pragma once
class OtsuOpenMPBinarizer
{
public:
	static PngImage* binarize(PngImage* imageToBinarize, int cpuThreads);
private:
	OtsuOpenMPBinarizer();
	static void calculateHistogram(const std::vector<unsigned char>& image, std::vector<double>& histogram, int cpuThreads);
	static int findThreshold(std::vector<double>& histogram, long int totalPixels, int cpuThreads);
	static PngImage* binarizeImage(PngImage * imageToBinarize, int threshold, int cpuThreads);
	static void showHistogram(double* histogram);
};

