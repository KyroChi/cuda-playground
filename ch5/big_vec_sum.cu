/**
 * Do GPU compute on more blocks / threads than are available.
 */

#include "../common/book.h"

#define N ( 512 * 1024 )

void
add ( int* a, int* b, int* c )
{
	int tid = 0;
	while ( tid < N ) {
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}

__global__ void
add_cuda ( int* a, int* b, int* c )
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while ( tid < N ) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int
main ( void )
{
	struct timespec start, finish;
	double elapsed;
	
	int a[N], b[N], c[N];

	for ( int ii = 0; ii < N; ii++ ) {
		a[ii] = -ii;
		b[ii] = ii * ii;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
	add(a, b, c);
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("Host took \t%.8f seconds\n", elapsed);

	int *dev_a, *dev_b, *dev_c;

	clock_gettime(CLOCK_MONOTONIC, &start);
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
				  N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b,
				  N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c,
				  N * sizeof(int) ) );

	// Copy a and b into device memory
	HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
				  cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
				  cudaMemcpyHostToDevice ) );

	add_cuda<<<(N+127)/128, 256>>>( dev_a, dev_b, dev_c );

	HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
				  cudaMemcpyDeviceToHost ) );

	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("Device took \t%.8f seconds\n", elapsed);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}