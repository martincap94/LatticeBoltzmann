#pragma once

#define GLM_ENABLE_EXPERIMENTAL


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
#define GRID_WIDTH 60
#define GRID_HEIGHT 60
#define GRID_DEPTH 40
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


#define NUM_PARTICLES 10000

#define CAMERA_VELOCITY 60.0f

#define TAU 0.9f
#define ITAU (1.0f / TAU)

using namespace std;
