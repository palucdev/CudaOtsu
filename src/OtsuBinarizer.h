#include "model/PngImage.h" 

#pragma once
class OtsuBinarizer
{
public:
	static PngImage* binarize(PngImage* imageToBinarize);
private:
	OtsuBinarizer();
	static void calculateHistogram(std::vector<unsigned char>& image, std::vector<double>& histogram);
	static int findThreshold(std::vector<double>& histogram, long int totalPixels);
	static PngImage* binarizeImage(PngImage * imageToBinarize, int threshold);
	static void showHistogram(double* histogram);
};

