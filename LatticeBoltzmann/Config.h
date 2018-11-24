#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime.h>


//#define DRAW_VELOCITY_ARROWS
//#define DRAW_PARTICLE_VELOCITY_ARROWS

#define COLLIDER_FILENAME "512x256_01.ppm"
#define HEIGHTMAP_FILENAME "60x40_02.ppm"

#define RUN_LBM3D
//#define RUN_LBM2D
#define USE_REINDEXED_LBM2D
#define USE_REINDEXED_LBM3D

#define USE_CUDA

#define MIRROR_SIDES

#define CONFIG_FILE "config.txt"


#ifdef RUN_LBM3D
#define GRID_WIDTH 189
#define GRID_HEIGHT 123
#define GRID_DEPTH 41
#endif
#ifdef RUN_LBM2D
#define GRID_WIDTH 512
#define GRID_HEIGHT 256
#define GRID_DEPTH 1
#endif

#ifdef USE_CUDA
	#define USE_INTEROP
#endif

#define BLOCK_DIM 256

#define MAX_STREAMLINE_LENGTH 200


#define LAT_SPEED 1.0f
#define LAT_SPEED_SQ (LAT_SPEED * LAT_SPEED)


#define CAMERA_VELOCITY 60.0f

#define TAU 0.55f
#define ITAU (1.0f / TAU)

//#define SUBGRID_EXPERIMENTAL
#define SMAG_C 0.3f

using namespace std;
