#include <stdio.h>
#include <sys/time.h>
#include "lodepng.h"

float timedifference_msec(struct timeval t0, struct timeval t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

#define MAX(a, b) (a > b ? a : b)
#define MAX4(a, b, c, d) (MAX(MAX(a, b), MAX(c, d)))

__global__ void the_pool(unsigned char * d_out, unsigned char * d_in, unsigned width) {
	int i = 64 * blockIdx.x + 2 * threadIdx.x;
	int j = 64 * blockIdx.y + 2 * threadIdx.y;

	unsigned char val00, val01, val10, val11;

	for (int comp = 0; comp < 4; comp++) {
		val00 = d_in[4 * width * i + 4 * j + comp];
		val01 = d_in[4 * width * i + 4 * (j + 1) + comp];
		val10 = d_in[4 * width * (i + 1) + 4 * j + comp];
		val11 = d_in[4 * width * (i + 1) + 4 * (j + 1) + comp];

		unsigned char max_value = MAX4(val00, val01, val10, val11);
		d_out[1 * width * i + 2 * j + comp] = max_value;
	}

}

void process(char* input_filename, char* output_filename) {
	unsigned error;
	unsigned char *image, *new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	struct timeval stop, start, start_no_copy, stop_no_copy;
	gettimeofday(&start, NULL);

	long int size = width * height * sizeof(unsigned char) * 4;
	new_image = (unsigned char*) malloc(size);

	// printf("Loaded image with width %d and height %d. Random one is %d\n", width, height, image[4* 200 * width + 4 * 100 + 0]);

	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size / 4);

	// transfer the array to the GPU
	cudaMemcpy(d_in, image, size, cudaMemcpyHostToDevice);
	gettimeofday(&start_no_copy, NULL);

	// launch the kernel
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(width / 64, height / 64, 1);
	the_pool<<<dimGrid, dimBlock>>>(d_out, d_in, width);

	gettimeofday(&stop_no_copy, NULL);
	float elapsed_no_copy = timedifference_msec(start_no_copy, stop_no_copy);
	printf("GPU processing took %f ms\n", elapsed_no_copy);
	// copy back the result array to the CPU
	cudaMemcpy(new_image, d_out, size / 4, cudaMemcpyDeviceToHost);

	// Do the remainder
	int remainder_width = width % 64;
	int remainder_height = height % 64;

	for (int i = height - remainder_height; i < height -1; i+=2) {
		for (int j = 0; j < width - 1; j+=2) {
			unsigned char val00, val01, val10, val11;

			for (int comp = 0; comp < 4; comp++) {
				val00 = image[4 * width * i + 4 * j + comp];
				val01 = image[4 * width * i + 4 * (j + 1) + comp];
				val10 = image[4 * width * (i + 1) + 4 * j + comp];
				val11 = image[4 * width * (i + 1) + 4 * (j + 1) + comp];

				unsigned char max_value = MAX4(val00, val01, val10, val11);
				new_image[1 * width * i + 2 * j + comp] = max_value;
			}
		}
	}

	for (int i = 0; i < height -1; i+=2) {
		for (int j = width - remainder_width; j < width - 1; j+=2) {
			unsigned char val00, val01, val10, val11;

			for (int comp = 0; comp < 4; comp++) {
				val00 = image[4 * width * i + 4 * j + comp];
				val01 = image[4 * width * i + 4 * (j + 1) + comp];
				val10 = image[4 * width * (i + 1) + 4 * j + comp];
				val11 = image[4 * width * (i + 1) + 4 * (j + 1) + comp];

				unsigned char max_value = MAX4(val00, val01, val10, val11);
				new_image[1 * width * i + 2 * j + comp] = max_value;
			}
		}
	}

	gettimeofday(&stop, NULL);
	float elapsed = timedifference_msec(start, stop);
	printf("Pool took %f ms\n", elapsed);

	lodepng_encode32_file(output_filename, new_image, width/2, height/2);

	free(image);
	free(new_image);
}

int main(int argc, char ** argv) {
	char* input_filename = argv[1];
	char* output_filename = argv[2];

	process(input_filename, output_filename);

	return 0;
}
