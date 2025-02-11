#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <string.h>

#define FILTER_RADIUS 2
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

enum Method {
    SIMPLE,
    GLOBAL
};

unsigned char* imageToArray(const char* filePath, int* width, int* height, int* channels) {
    printf("started image conversion\n");
    cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error loading image \n");
        return NULL;
    }
    *height = img.rows;
    *width = img.cols;
    *channels = img.channels(); // should be 3 for ALL color images
    printf("channels %d\n", *channels);

    int size = (*height) * (*width) * (*channels) * sizeof(unsigned char);
    unsigned char* imgArr = (unsigned char*) malloc(size);
    memcpy(imgArr, img.data, size);

    return imgArr;
}

void saveImageFromData(const char* filePath, unsigned char* dataArray, int width, int height, int channels) {
    if (dataArray == NULL || (channels != 1 && channels != 3)) {
        fprintf(stderr, "Invalid input data or unsupported channel count (only 1 or 3 channels supported)\n");
        return;
    }

    // Create a cv::Mat based on the channel count
    cv::Mat image;
    if (channels == 3) {
        // RGB image (3 channels)
        image = cv::Mat(height, width, CV_8UC3, dataArray);
    } else if (channels == 1) {
        // Grayscale image (1 channel)
        image = cv::Mat(height, width, CV_8UC1, dataArray);
    }

    // Save the image using OpenCV's imwrite function
    if (!cv::imwrite(filePath, image)) {
        fprintf(stderr, "Error saving image to %s\n", filePath);
    }
}

__global__
void crossCorrelationKernel(unsigned char* N, float* f, int* P, int height, int width, int channels, int r) {
    // poor memory coalescing 

    //assumes even number of filter_size
    int chan = blockIdx.z * blockDim.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Pval = 0;

    for (int fRow = 0; fRow < 2*r + 1; fRow++) {
        for (int fCol = 0; fCol < 2*r + 1; fCol++) {
            int inCol = col - r + fCol;
            int inRow = row - r + fRow;
            if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
                Pval += N[inRow * (channels * width) + inCol * (channels) + chan] * f[fCol * (2*r + 1) + fRow];
            }
        }
    }
    P[row * (channels * width) + col * (channels) + chan] = Pval;
}

__global__
void constMemCrossCorrelationKernel(unsigned char* N, int* P, int height, int width, int channels, int r) {
    // poor memory coalescing 

    //assumes even number of filter_size
    int chan = blockIdx.z * blockDim.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Pval = 0;

    for (int fRow = 0; fRow < 2*r + 1; fRow++) {
        for (int fCol = 0; fCol < 2*r + 1; fCol++) {
            int inCol = col - r + fCol;
            int inRow = row - r + fRow;
            if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
                Pval += N[inRow * (channels * width) + inCol * (channels) + chan] * F[fRow][fCol];
            }
        }
    }
    P[row * (channels * width) + col * (channels) + chan] = Pval;
}

void crossCorrelation(unsigned char* N, float* k, unsigned char* P, int height, int width, int channels, int r, Method method) {
    // assumes filter is single channel and applies independently across input channels
    float maxFilterValue = 0.0f;
    int filterSize = (2*r + 1) * (2*r + 1);
    for (int i = 0; i < filterSize; i++) {
        if (k[i] > maxFilterValue) {
            maxFilterValue = k[i];
        }
    }

    if (maxFilterValue > 1.0f) {
        for (int i = 0; i < filterSize; i++) {
            k[i] /= maxFilterValue;
        }
    }

    unsigned char* N_d;
    int* P_d;
    float* k_d; // ensure filter has max value of 1
    cudaMalloc((void **) &N_d, height * width * channels * sizeof(unsigned char));
    cudaMalloc((void **) &P_d, height * width * channels * sizeof(int)); // ensure 
    cudaMalloc((void **) &k_d, r * r * sizeof(float));

    cudaMemcpy(N_d, N, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k, r * r * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    switch (method) {
        case SIMPLE: {
            int threadCount = 16;
            dim3 gridDim(ceil(1.0 * height / threadCount), ceil(1.0 * width / threadCount), channels);
            dim3 blockDim(threadCount, threadCount, 1); //

            cudaEventRecord(start, 0);
            crossCorrelationKernel<<<gridDim, blockDim>>>(N_d, k_d, P_d, height, width, channels, r);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("SIMPLE kernel execution time: %f ms\n", elapsedTime);
            break;
        }
        case GLOBAL: {
            cudaMemcpyToSymbol(F, k, (2 * r + 1) * (2 * r + 1) * sizeof(float));
            int threadCount = 16;
            dim3 gridDim(ceil(1.0 * height / threadCount), ceil(1.0 * width / threadCount), channels);
            dim3 blockDim(threadCount, threadCount, 1);
            
            cudaEventRecord(start, 0);
            constMemCrossCorrelationKernel<<<gridDim, blockDim>>>(N_d, P_d, height, width, channels, r);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("GLOBAL kernel execution time: %f ms\n", elapsedTime);
            break;
        }
    }

    int* temp_P = new int[height * width * channels];
    cudaMemcpy(temp_P, P_d, height * width * channels * sizeof(int), cudaMemcpyDeviceToHost);

    int maxVal = temp_P[0];
    int minVal = temp_P[0];
    for (int i = 1; i < height * width * channels; i++) {
        if (temp_P[i] > maxVal) {
            maxVal = temp_P[i];
        }
        if (temp_P[i] < minVal) {
            minVal = temp_P[i];
        }
    }
    
    for (int i = 0; i < height * width * channels; i++) {
        P[i] = (temp_P[i] - minVal) * 255 / (maxVal - minVal);
    }
    
    delete[] temp_P;

    cudaFree(N_d);
    cudaFree(P_d);
    cudaFree(k_d);
}

int main(int argc, char** argv) {
    // data prep (e.g. import an image)
    printf("started main\n");
    int height, width, channels;
    char* sourcePath = "fake_img.jpg";
    unsigned char* imgArr = imageToArray("fake_img.jpg", &width, &height, &channels);
    printf("height: %d, width: %d\n", height, width);
    unsigned char* transformedImgArr= new unsigned char[height * width * channels];

    // create filter
    int filter_radius = FILTER_RADIUS;
    int filter_size = 2 * filter_radius + 1;
    float* filter = new float[filter_size * filter_size];
    for (int i = 0; i < filter_size * filter_size; i++) {    
        filter[i] = 1;
    }

    for (Method method = Method::SIMPLE; method <= Method::GLOBAL; method = static_cast<Method>(method + 1)) {
        crossCorrelation(imgArr, filter, transformedImgArr, height, width, channels, filter_radius, method);
    }

    char* savePath = new char[256];
    sprintf(savePath, "transformed_%s", sourcePath);
    saveImageFromData(savePath, transformedImgArr, width, height, 3);

    delete[] imgArr;
    delete[] transformedImgArr;
    delete[] filter;
}