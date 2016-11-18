#include <stdio.h>

__global__ void square(float * d_out, float * d_in){
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	printf("thread %d in block %d: idx = %d\n", threadIdx.x, blockIdx.x, idx);
	float f = d_in[idx];
	d_out[idx] = f*f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 10;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc(&d_in, ARRAY_BYTES);
	cudaMalloc(&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	dim3 dimBlock(ARRAY_SIZE/2, 1, 1);
	dim3 dimGrid(2, 1, 1);
	square<<<dimGrid, dimBlock>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	printf("\n");

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
