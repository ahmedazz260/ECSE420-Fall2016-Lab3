#include <stdio.h>
#include <sys/time.h>
#include "lodepng.h"

float timedifference_msec(struct timeval t0, struct timeval t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

__global__ void the_convolution(unsigned char * d_out, unsigned char * d_in, unsigned width) {
	const float w[3][3] = {
		1,2,-1,
		2,0.25,-2,
		1,-2,-1
	};

	int i = 32 * blockIdx.x + 1 * threadIdx.x + 1;
	int j = 32 * blockIdx.y + 1 * threadIdx.y + 1;

	unsigned new_width = width - 2;

	for (int comp = 0; comp < 3; comp++) {
		short sum = 0;
		for (int u = -1; u < 2; u++) {
			for (int v = -1; v < 2; v++) {
				sum += d_in[4 * width * (i + u) + 4 * (j + v) + comp] * w[u + 1][v + 1];
			}
		}

		sum = sum < 0 ? 0 : sum;
		sum = sum > 255 ? 255 : sum;
		d_out[4 * new_width * (i-1) + 4 * (j-1) + comp] = (unsigned char) sum;
	}

	d_out[4 * new_width * (i-1) + 4 * (j-1) + 3] = 255;
}

void process(char* input_filename, char* output_filename) {
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height, new_width, new_height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	struct timeval stop, start, start_no_copy, stop_no_copy;
	gettimeofday(&start, NULL);

	new_width = width - 2;
	new_height = height - 2;
	long int size = width * height * sizeof(unsigned char) * 4;
	long int new_size = new_width * new_height * sizeof(unsigned char) * 4;
	new_image = (unsigned char*) malloc(size);

	// printf("Loaded image with width %d and height %d.\n", width, height);

	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, new_size);

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);
	gettimeofday(&start_no_copy, NULL);

	// launch the kernel
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(new_width / 32, new_height / 32, 1);
	the_convolution<<<dimGrid, dimBlock>>>(d_out, d_in, width);

	gettimeofday(&stop_no_copy, NULL);
	float elapsed_no_copy = timedifference_msec(start_no_copy, stop_no_copy);
	printf("GPU processing took %f ms\n", elapsed_no_copy);

	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, new_size, cudaMemcpyDeviceToHost);

	// Do the remainder
	int remainder_width = width % 64;
	int remainder_height = height % 64;

	const float w[3][3] = {
		1,2,-1,
		2,0.25,-2,
		1,-2,-1
	};

	// printf("Remainder %d, %d.\n", width - remainder_width, height - remainder_height);
	for (int i = height - remainder_height; i < height -1; i++) {
		for (int j = 1; j < width - 1; j++) {
			for (int comp = 0; comp < 3; comp++) {
				short sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += image[4 * width * (i + u) + 4 * (j + v) + comp] * w[u + 1][v + 1];
					}
				}

				sum = sum < 0 ? 0 : sum;
				sum = sum > 255 ? 255 : sum;
				new_image[4 * new_width * (i-1) + 4 * (j-1) + comp] = (unsigned char) sum;
			}
		}
	}

	for (int i = 1; i < height -1; i++) {
		for (int j = width - remainder_width; j < width - 1; j++) {
			for (int comp = 0; comp < 3; comp++) {
				short sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += image[4 * width * (i + u) + 4 * (j + v) + comp] * w[u + 1][v + 1];
					}
				}

				sum = sum < 0 ? 0 : sum;
				sum = sum > 255 ? 255 : sum;
				new_image[4 * new_width * (i-1) + 4 * (j-1) + comp] = (unsigned char) sum;
			}
		}
	}

	gettimeofday(&stop, NULL);
	float elapsed = timedifference_msec(start, stop);
	printf("Convolve took %f ms\n", elapsed);

	lodepng_encode32_file(output_filename, new_image, new_width, new_height);

	free(image);
	free(new_image);
}

int main(int argc, char ** argv) {
	char* input_filename = argv[1];
	char* output_filename = argv[2];

	process(input_filename, output_filename);

	return 0;
}
