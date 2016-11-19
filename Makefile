CC = nvcc

all : rectify pool convolve

grid_4_4: grid_4_4.o
	$(CC) grid_4_4.o -o grid_4_4

rectify: lodepng.o rectify.o
	$(CC) lodepng.o rectify.o -o rectify

pool: lodepng.o pool.o
	$(CC) lodepng.o pool.o -o pool

convolve: lodepng.o convolve.o
	$(CC) lodepng.o convolve.o -o convolve




grid_4_4.o: grid_4_4.cu
	$(CC) -c grid_4_4.cu -o grid_4_4.o

rectify.o: rectify.cu
	$(CC) -c rectify.cu -o rectify.o

pool.o: pool.cu
	$(CC) -c pool.cu -o pool.o

convolve.o: convolve.cu
	$(CC) -c convolve.cu -o convolve.o

lodepng.o: lodepng.cu
	$(CC) -c lodepng.cu -o lodepng.o

clean:
	rm -f rectify.o
	rm -f rectify
	rm -f pool.o
	rm -f pool
	rm -f convolve.o
	rm -f convolve
	rm -f lodepng.o