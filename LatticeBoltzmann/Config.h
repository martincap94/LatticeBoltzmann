///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Config.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Configuration file.
*
*  Configuration header file that contains some of the variables that or not modifiable at runtime 
*  (but at compile time).
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime.h>


// NEEDS CLEANUP !!!

#define TEXTURES_DIR "textures/"

//#define LBM_EXPERIMENTAL // experimental features


//#define DRAW_VELOCITY_ARROWS
//#define DRAW_PARTICLE_VELOCITY_ARROWS

#define MIRROR_SIDES

#define CONFIG_FILE "config.ini"


#define BLOCK_DIM 256

#define MAX_STREAMLINE_LENGTH 200


#define LAT_SPEED 1.0f							// Experimental value, only used in 2D; do not modify for stable simulation
#define LAT_SPEED_SQ (LAT_SPEED * LAT_SPEED)	// Square of LAT_SPEED


#define DEFAULT_CAMERA_SPEED 60.0f

#define TAU 0.55f
#define ITAU (1.0f / TAU)

//#define SUBGRID_EXPERIMENTAL
#define SMAG_C 0.3f

#define VISUALIZE_VELOCITY

using namespace std;
