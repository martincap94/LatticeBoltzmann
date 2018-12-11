#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

/// Adapted from slides
static void handleError(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
		exit(EXIT_FAILURE);
	}
}
#define CHECK_ERROR( error ) ( handleError( error, __FILE__, __LINE__ ) )



