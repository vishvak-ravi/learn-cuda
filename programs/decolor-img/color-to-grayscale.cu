#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <string.h>
#include "misc.cuh"

#define CHANNELS 3

unsigned char* imageToArray(const char* filePath, int* width, int* height, int* channels) {
    printf("started image conversion\n");
    cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error loading image \n");
        return NULL;
    }
    *height = img.rows;
    *width = img.cols;
    *channels = img.channels();
    printf("channels %d\n", *channels);

    int size = (*height) * (*width) * (*channels) * sizeof(unsigned char);
    unsigned char* imgArr = (unsigned char*) malloc(size);
    memcpy(imgArr, img.data, size);

    return imgArr;
}

void saveImageFromData(const char* filePath, unsigned char* dataArray, int* width, int* height, int* channels) {
    if (dataArray == NULL || (*channels != 1 && *channels != 3)) {
        fprintf(stderr, "Invalid input data or unsupported channel count (only 1 or 3 channels supported)\n");
        return;
    }

    // Create a cv::Mat based on the channel count
    cv::Mat image;
    if (*channels == 3) {
        // RGB image (3 channels)
        image = cv::Mat(*height, *width, CV_8UC3, dataArray);
    } else if (*channels == 1) {
        // Grayscale image (1 channel)
        image = cv::Mat(*height, *width, CV_8UC1, dataArray);
    }

    // Save the image using OpenCV's imwrite function
    if (!cv::imwrite(filePath, image)) {
        fprintf(stderr, "Error saving image to %s\n", filePath);
    }
}

__global__
void rgbToGrayscaleKernel(unsigned char* P_in, unsigned char* P_out, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int grayOffset = row * width + col;
        int rgbOffset = CHANNELS * grayOffset;
        unsigned char r = P_in[rgbOffset];
        unsigned char g = P_in[rgbOffset + 1];
        unsigned char b = P_in[rgbOffset + 2];
        P_out[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void rgbToGrayscale(unsigned char* P_in_h, unsigned char* P_out_h, int height, int width) {
    unsigned char *P_in_d, *P_out_d;
    int size = height * width * 3;
    int outSize = height * width;
    cudaMalloc((void **) &P_in_d, size);
    cudaMalloc((void **) &P_out_d, outSize);

    cudaMemcpy(P_in_d, P_in_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_out_d, P_out_h, outSize, cudaMemcpyHostToDevice);

    int threadCount = 12;

    dim3 gridDim(ceil(1.0 * width / threadCount), ceil(1.0 * height / threadCount));
    dim3 blockDim(threadCount, threadCount, 1);

    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(P_in_d, P_out_d, height, width);

    cudaMemcpy(P_out_h, P_out_d, outSize, cudaMemcpyDeviceToHost);

    cudaFree(P_in_d);
    cudaFree(P_out_d);
}

int main(int argc, char** argv) {
    // data prep (e.g. import an image)
    printf("started main\n");
    int height, width, channels;
    char* sourcePath = "../fake_img.jpg";
    unsigned char* imgArr = imageToArray("../fake_img.jpg", &width, &height, &channels);
    printf("height: %d, width: %d\n", height, width);
    unsigned char* transformedImgArr= new unsigned char[height * width];

    rgbToGrayscale(imgArr, transformedImgArr, height, width);

    char* savePath = new char[256];
    sprintf(savePath, "transformed_fake_img.jpg", sourcePath);
    int grayChannels = 1;
    int* grayChannelsPtr = &grayChannels;
    saveImageFromData(savePath, transformedImgArr, &width, &height, grayChannelsPtr);

    delete[] imgArr;
    delete[] transformedImgArr;
}