#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

#define SIZE 4
#define TOTAL_SIZE (SIZE * SIZE)
#define TOTAL_BYTES (TOTAL_SIZE * sizeof(float))

#define GET_VAL(arr, x, y) arr[x * SIZE + y]
#define SET_VAL(arr, x, y, val) arr[x * SIZE + y] = val

__global__ void the_iteration(float * new_val, float * val1, float * val2) {
	int i = 64 * blockIdx.x + 2 * threadIdx.x;
	int j = 64 * blockIdx.y + 2 * threadIdx.y;
}

void process(int iteration_count) {
	float * initial_values;
	initial_values = (float *) malloc(sizeof(float) * TOTAL_SIZE);
	SET_VAL(initial_values, 2, 2, 1.0);

	printf("Iteration count is %d.\n", iteration_count);

	// declare GPU memory pointers
	float * new_val;
	float * val1;
	float * val2;

	// allocate GPU memory
	cudaMalloc(&new_val, TOTAL_BYTES);
	cudaMalloc(&val1, TOTAL_BYTES);
	cudaMalloc(&val2, TOTAL_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(val1, initial_values, TOTAL_BYTES, cudaMemcpyHostToDevice);

	dim3 dimBlock(4, 4, 1);
	dim3 dimGrid(1, 1, 1);
	for (int i = 0; i < 1; i++) {
		// launch the kernel
		the_iteration<<<dimGrid, dimBlock>>>(new_val, val1, val2);

		float * temp = val2;
		val2 = val1;
		val1 = new_val;
		new_val = temp; // Reuse the allocated memory.
	}

	// copy back the result array to the CPU
	cudaMemcpy(initial_values, val1, TOTAL_BYTES, cudaMemcpyDeviceToHost);

	printf("Value is %f\n", GET_VAL(initial_values, 2, 222));
}

int main(int argc, char ** argv) {
	int iteration_count = atoi(argv[1]);

	process(iteration_count);

	return 0;
}
