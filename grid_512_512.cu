#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

#define G 0.75
#define RHO 0.5
#define ETA 0.0002

#define CELLS_ROW_PER_THREAD 4 // CELLS_ROW_PER_THREAD ^ 2 cells per thread
#define THREADS_ROW_PER_BLOCK 32 // THREADS_ROW_PER_BLOCK ^ 2 threads per block
#define BLOCKS_ROW_PER_GRID 4 // BLOCKS_ROW_PER_GRID ^ 2 blocks per grid

#define CELLS_ROW_PER_BLOCK (CELLS_ROW_PER_THREAD * THREADS_ROW_PER_BLOCK)

#define SIZE 512
#define TOTAL_SIZE (SIZE * SIZE)
#define TOTAL_BYTES (TOTAL_SIZE * sizeof(float))

#define GET_VAL(arr, x, y) arr[(x) * SIZE + y]
#define SET_VAL(arr, x, y, val) arr[(x) * SIZE + y] = val

#define get_index(x, y) ((x) * SIZE + y)
#define get_row(index) (index / SIZE)
#define get_col(index) (index % SIZE)

#define is_inner(x, y) (x > 0 && x < SIZE - 1 && y > 0 && y < SIZE - 1)

#define GET_VAL_INDEX(arr, index) GET_VAL(arr, get_row(index), get_col(index))
#define SET_VAL_INDEX(arr, index, val) GET_VAL(arr, get_row(index), get_col(index)) = val

__global__ void the_block_iteration(float * new_val, float * val1, float * val2) {
	int start_row = CELLS_ROW_PER_BLOCK * blockIdx.x + CELLS_ROW_PER_THREAD * threadIdx.x;
	int start_col = CELLS_ROW_PER_BLOCK * blockIdx.y + CELLS_ROW_PER_THREAD * threadIdx.y;
	int end_row = start_row + CELLS_ROW_PER_THREAD; // Not inclusive
	int end_col = start_col + CELLS_ROW_PER_THREAD; // Not inclusive

	float value = 0;

	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			if (is_inner(row, col)) { // Process inner
				value = GET_VAL(val1, row - 1, col) + GET_VAL(val1, row + 1, col) + GET_VAL(val1, row, col + 1) + GET_VAL(val1, row, col - 1);
				value -= 4 * GET_VAL(val1, row, col);
				value *= RHO;
				value += 2 * GET_VAL(val1, row, col) - (1 - ETA) * GET_VAL(val2, row, col);
				value /= (1 + ETA);
				SET_VAL(new_val, row, col, value);
			}
		}
	}

	if (start_row == 0 || end_row == SIZE || start_col == 0 || end_col == SIZE) {
		__syncthreads();
	}

	if (start_row == 0) { // If contains top border
		int row = start_row;
		for (int col = start_col; col < end_col; col++) {
			value = G * GET_VAL(new_val, row + 1, col);
			SET_VAL(new_val, row, col, value);
		}
	} else if (end_row == SIZE) { // If contains bot border
		int row = end_row - 1;
		for (int col = start_col; col < end_col; col++) {
			value = G * GET_VAL(new_val, row - 1, col);
			SET_VAL(new_val, row, col, value);
		}
	} else if (start_col == 0) { // If contains left border
		int col = start_col;
		for (int row = start_row; row < end_row; row++) {
			value = G * GET_VAL(new_val, row, col + 1);
			SET_VAL(new_val, row, col, value);
		}
	} else if (end_col == SIZE) { // If contains right border
		int col = end_col - 1;
		for (int row = start_row; row < end_row; row++) {
			value = G * GET_VAL(new_val, row, col - 1);
			SET_VAL(new_val, row, col, value);
		}
	}

	if ((start_row == 0 && start_col == 0) ||
		(start_row == 0 && end_col == SIZE) ||
		(end_row == SIZE && start_col == 0) ||
		(end_row == SIZE && end_col == SIZE)) {
		__syncthreads();
	}

	if (start_row == 0 && start_col == 0) { // Top left corner
		value = G * GET_VAL(new_val, 0 + 1, 0);
		SET_VAL(new_val, 0, 0, value);
	} else if (start_row == 0 && end_col == SIZE) { // Top right corner
		value = G * GET_VAL(new_val, 0, SIZE - 2);
		SET_VAL(new_val, 0, SIZE - 1, value);
	} else if (end_row == SIZE && start_col == 0) { // Bot left corner
		value = G * GET_VAL(new_val, SIZE - 2, 0);
		SET_VAL(new_val, SIZE - 1, 0, value);
	} else if (end_row == SIZE && end_col == SIZE) { // Bot right corner
		value = G * GET_VAL(new_val, SIZE - 1, SIZE - 2);
		SET_VAL(new_val, SIZE - 1, SIZE - 1, value);
	}


	if (start_row == 0 && start_col == 0) {
		printf("%f,\n", GET_VAL(new_val, SIZE / 2, SIZE / 2));
	}
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
	SET_VAL(initial_values, SIZE / 2, SIZE / 2, 1.0);
	cudaMemcpy(val1, initial_values, TOTAL_BYTES, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_ROW_PER_BLOCK, THREADS_ROW_PER_BLOCK, 1);
	dim3 dimGrid(BLOCKS_ROW_PER_GRID, BLOCKS_ROW_PER_GRID, 1);
	for (int i = 0; i < iteration_count; i++) {
		// launch the kernel
		the_block_iteration<<<dimGrid, dimBlock>>>(new_val, val1, val2);

		float * temp = val2;
		val2 = val1;
		val1 = new_val;
		new_val = temp; // Reuse the allocated memory.
	}

	// copy back the result array to the CPU
	cudaMemcpy(initial_values, val1, TOTAL_BYTES, cudaMemcpyDeviceToHost);

	// printf("%f,\n", GET_VAL(initial_values, SIZE / 2, SIZE / 2));
}

int main(int argc, char ** argv) {
	int iteration_count = atoi(argv[1]);

	process(iteration_count);

	return 0;
}
