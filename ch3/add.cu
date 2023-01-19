#include <stdio.h>
#include "../common/book.h"

__global__ void
add ( int a, int b, int* c )
{
	*c = a + b;
}

int
main ( void )
{
	int a = 2, b = 7;
	int c, *dev_c;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

	add<<<1,1>>>( a, b, dev_c );

	HANDLE_ERROR( cudaMemcpy( &c,
				  dev_c,
				  sizeof(int),
				  cudaMemcpyDeviceToHost ) );

	printf("%d + %d = %d\n", a, b, c);
	cudaFree(dev_c);

	return 0;
}