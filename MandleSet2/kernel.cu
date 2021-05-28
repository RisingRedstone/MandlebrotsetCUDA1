#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <time.h>
#include <cmath>

#define Height 800
#define Width 800
#define batch 400
#define trials 400
#define rate 0.05
#define ReCoord -0.761574
#define ImCoord -0.0847596
#define Intensity 0.5
#define Speed 1
#define debug false

using namespace cv;
using namespace std;

typedef struct {
	double* N;
	double* I;
	double* posN;
	double* posI;
}Com;

double mapp(double i, double Mid, double Range, int size) {
	return (((i / double(size - 1)) - 0.5) * Range) + Mid;
}

double func(double i, int size, double offset, double magnification) {
	return (((i / double(size - 1)) + offset) / magnification);
}

__global__ void calculate(Com C, uint8_t* Conf) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	C.I[index] = 0;
	C.N[index] = 0;
	double tempN = C.N[index];
	double sqrMagN = tempN * tempN;
	double sqrMagI = C.I[index] * C.I[index];

	Conf[index] = 255;

	int i = 0;
	for (i = 0; i < trials; i++) {
		tempN = C.N[index];

		C.N[index] = (sqrMagN - sqrMagI) + C.posN[index];
		C.I[index] = (2 * C.I[index] * tempN) + C.posI[index];

		sqrMagN = (C.N[index] * C.N[index]);
		sqrMagI = (C.I[index] * C.I[index]);

		if ((sqrMagN + sqrMagI) >= 4) {
			if (i * Intensity < 256)
				Conf[index] = (uint8_t)(i * Intensity);
			else
				Conf[index] = 255;
			break;
		}
	}
}

__global__ void initialize(Com C, int bat, double iteration) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	C.I[index] = 0;
	C.N[index] = 0;
	C.posN[index] = ((((double)threadIdx.x / double(Width - 1)) - 0.5) / iteration) + ReCoord;
	C.posI[index] = ((((double)(bat * batch + blockIdx.x) / double(Height - 1)) - 0.5) / iteration) + ImCoord;

}

void checkDevice() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
}

int main()
{
	checkDevice();

	//Image Variable
	Mat image = Mat::zeros(Height, Width, CV_8UC1);

	//Com C;
	Com c_C;
	uint8_t* Conf;
	uint8_t* c_Conf;

	size_t size = sizeof(double) * Height * batch;

	//C.I = (double*)malloc(size);
	//C.N = (double*)malloc(size);
	Conf = (uint8_t*)malloc(sizeof(uint8_t) * Height * batch);
	cudaMalloc(&c_C.I, size);
	cudaMalloc(&c_C.N, size);
	cudaMalloc(&c_C.posI, size);
	cudaMalloc(&c_C.posN, size);
	cudaMalloc(&c_Conf, sizeof(uint8_t) * Height * batch);

	int z = 0;
	int iter = 0;

	while (!debug) {
		for (z = 0; z < (Width / batch); z+=Speed) {
			int i = 0;
			int j = 0;

			initialize << <batch, Height >> > (c_C, z, (double)pow(2, (rate * (double)iter)));

			cudaDeviceSynchronize();
			
			cudaMemcpy(c_Conf, Conf, sizeof(uint8_t) * Height * batch, cudaMemcpyHostToDevice);

			calculate << <batch, Height >> > (c_C, c_Conf);

			cudaDeviceSynchronize();

			//cudaMemcpy(C.I, c_C.I, size, cudaMemcpyDeviceToHost);
			//cudaMemcpy(C.N, c_C.N, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(Conf, c_Conf, sizeof(uint8_t) * Height * batch, cudaMemcpyDeviceToHost);

			for (j = 0; j < batch; j++) {
				for (i = 0; i < Height; i++) {
					image.data[((z * batch + j) * Height + i)] = Conf[(j * Height + i)];
				}
			}
		}
		
		
		imshow("Display Window", image);
		char c = (char)waitKey(25);
		if (c == 27)
			break;
		

		iter++;
	}

	cudaFree(c_C.I);
	cudaFree(c_C.N);
	cudaFree(c_C.posI);
	cudaFree(c_C.posN);
	cudaFree(c_Conf);
	free(Conf);
	//free(C.I);
	//free(C.N);
	//free(C.posI);
	//free(C.posN);


	/*
	imshow("Display Window", image);
	waitKey(0);
	*/

	return 0;
}