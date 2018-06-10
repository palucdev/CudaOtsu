# CudaOtsu
Otsu's method thresholding and image binarization on GPU in CUDA in C++.

## Otsu's Method
Simple and one of the most popular image thresholding method used in computer vision problems. Algorithm helps to find optimal threshold value for greyscale image to be then used in image binarization. As binarized images are main data type for problems such as OCR, Otsu's method is used as a part of pre-processing pipelines for computer vision problems.

Otsu's main idea is to find threshold that minimzes the intra-class variance within 'foreground' and 'background' classes. Based on observation that minimazing intra-class variance is the same as maximizing inter-class variance, we can define algorithm as following steps:

1. Compute histogram and probabilities of each intensity level
2. Set up initial classes probability and classes mean
3. For every intensity level:
    - Update class probability and mean for current intensity level
    - Compute inter-class variance
4. Threshold will be at intenisty level with highest inter-class variance

### Otsu's Method Visualization
![Otsu's Method Visualization](https://upload.wikimedia.org/wikipedia/commons/3/34/Otsu%27s_Method_Visualization.gif)

## Project goal
Project is focused on implementing Otsu's method as CUDA kernels to test how well GPU will handle this algorithm in terms of computation time (comparing to multithreaded CPU implementation). Also it's a kind of CUDA playground for me as I'm absolute beginner in GPU computations. 

## Current features 
- C++ implementation of Otsu's method on CPU (single threaded)
- Basic CUDA implementation of Otsu's method on GPU
- Basic CUDA shared memory usage (no huge speed boost here, Otsu's algorithm gains very little from cache)
- Makefile for more multiplatform approach
- Extendable Binarizers architecture

## To do
- [ ] Concurrent CPU method implementation for benchmarking (openMP)
- [ ] Memory optimization
- [ ] Simple (cross-platform) GUI 
- [ ] Cuda GPU selection error handling
- [ ] Research and compare different CUDA optimization mechanisms 
