#include "../common/book.h"

#define N 10

__global__ void
add ( int* a, int* b, int* c )
{
	int tid = threadIdx.x;
	if ( tid < N ) {
		c[tid] = a[tid] + b[tid];
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

	add<<<(N+127)/128, 128>>>( dev_a, dev_b, dev_c );

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