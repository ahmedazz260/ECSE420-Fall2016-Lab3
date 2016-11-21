#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

#define G 0.75
#define RHO 0.5
#define ETA 0.0002


#define SIZE 4
#define TOTAL_SIZE (SIZE * SIZE)
#define TOTAL_BYTES (TOTAL_SIZE * sizeof(float))

#define GET_VAL(arr, x, y) arr[(x) * SIZE + y]
#define SET_VAL(arr, x, y, val) arr[(x) * SIZE + y] = val

#define get_index(x, y) ((x) * SIZE + y)
#define get_row(index) (index / SIZE)
#define get_col(index) (index % SIZE)

#define GET_VAL_INDEX(arr, index) GET_VAL(arr, get_row(index), get_col(index))
#define SET_VAL_INDEX(arr, index, val) GET_VAL(arr, get_row(index), get_col(index)) = val

__global__ void the_iteration(float * new_val, float * val1, float * val2) {
	int row = threadIdx.x;
	int col = threadIdx.y;

	int index = get_index(row, col);
	float value = 0;

	if (index == 5 || index == 6 || index == 9 || index == 10) {
		value = GET_VAL(val1, row - 1, col) + GET_VAL(val1, row + 1, col) + GET_VAL(val1, row, col + 1) + GET_VAL(val1, row, col - 1);
		value -= 4 * GET_VAL(val1, row, col);
		value *= RHO;
		value += 2 * GET_VAL(val1, row, col) - (1 - ETA) * GET_VAL(val2, row, col);
		value /= (1 + ETA);
	}

	SET_VAL_INDEX(new_val, index, value);
	__syncthreads();

	if (index == 1 || index == 2) { // Top border
		value = G * GET_VAL(new_val, row + 1, col);
	} else if (index == 4 || index == 8) { // Left border
		value = G * GET_VAL(new_val, row, col + 1);
	} else if (index == 7 || index == 11) { // Right border
		value = G * GET_VAL(new_val, row, col - 1);
	} else if (index == 13 || index == 14) { // Bottom border
		value = G * GET_VAL(new_val, row - 1, col);
	}

	SET_VAL_INDEX(new_val, index, value);
	__syncthreads();

	if (index == 0) {
		value = G * GET_VAL(new_val, row + 1, col);
	} else if (index == 3) {
		value = G * GET_VAL(new_val, row, col - 1);
	} else if (index == 12) {
		value = G * GET_VAL(new_val, row - 1, col);
	} else if (index == 15) {
		value = G * GET_VAL(new_val, row, col - 1);
	}

	SET_VAL_INDEX(new_val, index, value);

	// TODO ask Loren
	// if (row == 2 && col == 2) {
	// 	printf("%f,\n", GET_VAL(new_val, 32, 2));
	// }
}

void process(int iteration_count) {
	float * initial_values;
	initial_values = (float *) malloc(sizeof(float) * TOTAL_SIZE);
	for (int i = 0; i < TOTAL_SIZE; i++) {
		initial_values[i] = 0.0;
	}
	
	// printf("Iteration count is %d.\n", iteration_count);

	// declare GPU memory pointers
	float * new_val;
	float * val1;
	float * val2;

	// allocate GPU memory
	cudaMalloc(&new_val, TOTAL_BYTES);
	cudaMalloc(&val1, TOTAL_BYTES);
	cudaMalloc(&val2, TOTAL_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(new_val, initial_values, TOTAL_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(val2, initial_values, TOTAL_BYTES, cudaMemcpyHostToDevice);
	SET_VAL(initial_values, 2, 2, 1.0);
	cudaMemcpy(val1, initial_values, TOTAL_BYTES, cudaMemcpyHostToDevice);

	dim3 dimBlock(4, 4, 1);
	dim3 dimGrid(1, 1, 1);
	for (int i = 0; i < iteration_count; i++) {
		// launch the kernel
		the_iteration<<<dimGrid, dimBlock>>>(new_val, val1, val2);

		float * temp = val2;
		val2 = val1;
		val1 = new_val;
		new_val = temp; // Reuse the allocated memory.
	}

	// copy back the result array to the CPU
	cudaMemcpy(initial_values, val1, TOTAL_BYTES, cudaMemcpyDeviceToHost);
	
	// printf("%f,\n", GET_VAL(initial_values, 2, 2));
}

int main(int argc, char ** argv) {
	int iteration_count = atoi(argv[1]);

	process(iteration_count);

	return 0;
}
