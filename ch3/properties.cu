#include "../common/book.h"

int
main ( void )
{
	cudaDeviceProp prop;
	int count;
	
	HANDLE_ERROR( cudaGetDeviceCount( &count ) );
	for ( int ii = 0; ii < count; ii++ ) {
		HANDLE_ERROR( cudaGetDeviceProperties(&prop,
						      ii) );
		printf("   --- General INformation for device %d ---\n", ii);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major,
		       prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
	}

	return 0;
}