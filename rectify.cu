

#include <stdio.h>
#include "lodepng.h"

__global__ void rectify(unsigned char * d_output_image, unsigned char * d_input_image) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	d_output_image[4 * idx + 0] = d_input_image[4 * idx + 0] < 127 ? 127 : d_input_image[4 * idx + 0];
	d_output_image[4 * idx + 1] = d_input_image[4 * idx + 1] < 127 ? 127 : d_input_image[4 * idx + 1];
	d_output_image[4 * idx + 2] = d_input_image[4 * idx + 2] < 127 ? 127 : d_input_image[4 * idx + 2];
	d_output_image[4 * idx + 3] = 255;
}

int main(int argc, char** argv) {
	// get input args
	char * input_filename = argv[1];
	char * output_filename = argv[2];

	printf("Now rectifying to\n", input_filename);

	// load input image from png
	unsigned error;
	unsigned char * h_input_image, * h_output_image;
	unsigned width, height;

	error = lodepng_decode32_file(&h_input_image, &width, &height, input_filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
	
	const int IMAGE_SIZE = width * height;
	const int IMAGE_BYTES = IMAGE_SIZE * 4 * sizeof(unsigned char);

	h_output_image = (unsigned char *) malloc(IMAGE_BYTES);

	// declare GPU memory pointers
	unsigned char * d_input_image;
	unsigned char * d_output_image;

	// allocate GPU memory
	cudaMalloc(&d_input_image, IMAGE_BYTES);
	cudaMalloc(&d_output_image, IMAGE_BYTES);

	// transfer image to GPU
	cudaMemcpy(d_input_image, h_input_image, IMAGE_BYTES, cudaMemcpyHostToDevice);

	// launch kernel
	int block_size = 1024;
	rectify<<<block_size, IMAGE_SIZE / block_size>>>(d_output_image, d_input_image);
	int remainder = IMAGE_SIZE % block_size;
	cudaMemcpy(h_output_image, d_output_image, IMAGE_BYTES, cudaMemcpyDeviceToHost);

	// process the remainder on CPU
	for (int idx = IMAGE_SIZE - remainder; idx < IMAGE_SIZE; idx++) {
		h_output_image[4 * idx + 0] = h_input_image[4 * idx + 0] < 127 ? 127 : h_input_image[4 * idx + 0];
		h_output_image[4 * idx + 1] = h_input_image[4 * idx + 1] < 127 ? 127 : h_input_image[4 * idx + 1];
		h_output_image[4 * idx + 2] = h_input_image[4 * idx + 2] < 127 ? 127 : h_input_image[4 * idx + 2];
		h_output_image[4 * idx + 3] = 255;
	}

	lodepng_encode32_file(output_filename, h_output_image, width, height);

	cudaFree(d_input_image);
	cudaFree(d_output_image);
}