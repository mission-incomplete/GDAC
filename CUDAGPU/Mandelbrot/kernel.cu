#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include "config.h"
#include "util.h"

__device__ void set_pixel(unsigned char* image, int width, int x, int y, unsigned char* c)
{
	image[4 * width * y + 4 * x + 0] = c[0];
	image[4 * width * y + 4 * x + 1] = c[1];
	image[4 * width * y + 4 * x + 2] = c[2];
	image[4 * width * y + 4 * x + 3] = 255;
}

__global__ void kernel(unsigned char* image, unsigned char* colormap) 
{
	int row, col, index, iteration;
	double c_re, c_im, x, y, x_new;

	int width = WIDTH;
	int height = HEIGHT;
	int maxIterations = MAX_ITERATION;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	row = index / width;
	col = index % width;

	if (row >= height || col >= width || index >= LENGTH) 
	{
		return;
	}
	
	c_re = (col - width / 2.0) * 4.0 / width;
	c_im = (row - height / 2.0) * 4.0 / width;
	x = 0, y = 0;
	iteration = 0;
	while (x * x + y * y <= 4 && iteration < maxIterations) 
	{
		x_new = x * x - y * y + c_re;
		y = 2 * x * y + c_im;
		x = x_new;
		iteration++;
	}
	if (iteration > maxIterations) {
		iteration = maxIterations;
	}
	set_pixel(image, width, col, row, &colormap[iteration * 3]);
}

void onFail(const cudaError_t& err, const char message[])
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s, Error: %s\n" ,message ,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
int main(void)
{
	unsigned char* colormapCuda, *imageCuda;
	double times[REPEAT];
	struct timeb start, end;
	int i, r;
	char path[255];

	int colormapSize = (MAX_ITERATION + 1) * 3;
	int imageSize = LENGTH * 4;

	unsigned char* colormap = (unsigned char*)malloc(colormapSize);
	unsigned char* image = (unsigned char*)malloc(imageSize);

	init_colormap(MAX_ITERATION, colormap);

	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(0);
	onFail(err, "Failed to set device!");

	//int ndev;
	//cudaDeviceProp p;
	//err = cudaGetDeviceCount(&ndev);
	//onFail(err, "Failed to cudaGetDeviceCount!");
	//for (i = 0; i < ndev; i++)
	//{
	//	err = cudaGetDeviceProperties(&p, i);
	//	onFail(err, "Failed cudaGetDeviceProperties!");
	//	printf("Name: %s\n", p.name);
	//	printf("Compute capability: %d.%d\n", p.major, p.minor);
	//	printf("Max threads/block: %d\n", p.maxThreadsPerBlock);
	//	printf("Max block size: %d x %d x %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
	//	printf("Max grid size:  %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
	//	printf("Warp size: %d\n", p.warpSize);
	//}

	for (r = 0; r < REPEAT; r++)
	{
		memset(image, 0, imageSize);
		ftime(&start);

		err = cudaMalloc(&colormapCuda, colormapSize * sizeof(unsigned char));
		onFail(err, "Failed to cudaMalloc colormapCuda!");

		err = cudaMalloc(&imageCuda, imageSize * sizeof(unsigned char));
		onFail(err, "Failed to cudaMalloc imageCuda!");

		err = cudaMemcpy(colormapCuda, colormap, colormapSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		onFail(err, "Failed to cudaMemcpy hostToDevice colormapCuda!");

		err = cudaMemcpy(imageCuda, image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		onFail(err, "Failed to cudaMemcpy hostToDevice imageCuda!");

		std::cout << "kernel " << GRID_SIZE << ", " << BLOCK_SIZE << " LEN: " << LENGTH <<"\n";
		kernel << <GRID_SIZE, BLOCK_SIZE >> > (imageCuda, colormapCuda);
		
		err = cudaGetLastError();
		onFail(err, "Failed to launch kernel!");

		err = cudaDeviceSynchronize();
		onFail(err, "Failed to synchronize");

		err = cudaMemcpy(image, imageCuda, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		onFail(err, "Failed to cudaMemcpy deviceToHost imageCuda!");

		cudaFree(colormapCuda);
		cudaFree(imageCuda);

		ftime(&end);
		times[r] = end.time - start.time + ((double)end.millitm - (double)start.millitm) / 1000.0;

		sprintf(path, IMAGE, "gpu", r);
		save_image(path, image, WIDTH, HEIGHT);
		progress("gpu", r, times[r]);
	}
	report("gpu", times);

	err = cudaDeviceReset();
	onFail(err, "Failed to cudaDeviceReset!");

	free(image);
	free(colormap);
	cudaFree(colormapCuda);
	cudaFree(imageCuda);
	return 0;
}