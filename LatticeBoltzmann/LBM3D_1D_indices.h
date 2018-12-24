///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       LBM3D.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines the LBM3D class and CUDA constants that are used in 3D simulation on GPU.
*
*  Defines the LBM3D class (subclass of LBM) that implements the 3D simulation using CPU and GPU.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"
#include "ParticleSystem.h"
#include "DataStructures.h"
#include "HeightMap.h"

#include <cuda_gl_interop.h>


#include <vector>

#include "LBM.h"

__constant__ glm::vec3 dirVectorsConst[19];
__constant__ float WEIGHT_MIDDLE;
__constant__ float WEIGHT_AXIS;
__constant__ float WEIGHT_NON_AXIAL;


/// 3D LBM simulator.
/**
	3D LBM simulator that supports both CPU and GPU simulations.
	GPU (CUDA) simulation is run through global kernels that are defined in LBM3D_1D_indices.cu.
	The LBM is indexed as a 1D array in this implementation.
	The simulator supports particle velocity visualization.
*/
class LBM3D_1D_indices : public LBM {


	const glm::vec3 vMiddle = glm::vec3(0.0f, 0.0f, 0.0f);
	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBack = glm::vec3(0.0f, 0.0f, -1.0f);
	const glm::vec3 vFront = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vBackRight = glm::vec3(1.0f, 0.0f, -1.0f);
	const glm::vec3 vBackLeft = glm::vec3(-1.0f, 0.0f, -1.0f);
	const glm::vec3 vFrontRight = glm::vec3(1.0f, 0.0f, 1.0f);
	const glm::vec3 vFrontLeft = glm::vec3(-1.0f, 0.0f, 1.0f);
	const glm::vec3 vTopBack = glm::vec3(0.0f, 1.0f, -1.0f);
	const glm::vec3 vTopFront = glm::vec3(0.0f, 1.0f, 1.0f);
	const glm::vec3 vBottomBack = glm::vec3(0.0f, -1.0f, -1.0f);
	const glm::vec3 vBottomFront = glm::vec3(0.0f, -1.0f, 1.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);

	//const glm::vec3 directionVectors3D[19] = {
	//	glm::vec3(0.0f, 0.0f, 0.0f),
	//	glm::vec3(1.0f, 0.0f, 0.0f),
	//	glm::vec3(-1.0f, 0.0f, 0.0f),
	//	glm::vec3(0.0f, 0.0f, -1.0f),
	//	glm::vec3(0.0f, 0.0f, 1.0f),
	//	glm::vec3(0.0f, 1.0f, 0.0f),
	//	glm::vec3(0.0f, -1.0f, 0.0f),
	//	glm::vec3(1.0f, 0.0f, -1.0f),
	//	glm::vec3(-1.0f, 0.0f, -1.0f),
	//	glm::vec3(1.0f, 0.0f, 1.0f),
	//	glm::vec3(-1.0f, 0.0f, 1.0f),
	//	glm::vec3(0.0f, 1.0f, -1.0f),
	//	glm::vec3(0.0f, 1.0f, 1.0f),
	//	glm::vec3(0.0f, -1.0f, -1.0f),
	//	glm::vec3(0.0f, -1.0f, 1.0f),
	//	glm::vec3(1.0f, 1.0f, 0.0f),
	//	glm::vec3(-1.0f, 1.0f, 0.0f),
	//	glm::vec3(1.0f, -1.0f, 0.0f),
	//	glm::vec3(-1.0f, -1.0f, 0.0f)
	//};



public:


	Node3D *frontLattice;			///< Front lattice - the one currently drawn at end of each frame
	Node3D *backLattice;			///< Back lattice - the one to which we prepare next frame to be drawn

	Node3D *d_frontLattice;			///< Device pointer for the front lattice
	Node3D *d_backLattice;			///< Device pointer for the back lattice

	glm::vec3 *velocities;

	glm::vec3 *d_velocities;


	HeightMap *heightMap;
	float *d_heightMap;


	struct cudaGraphicsResource *cudaParticleVerticesVBO;
	struct cudaGraphicsResource *cudaParticleColorsVBO;


	LBM3D_1D_indices();
	LBM3D_1D_indices(glm::vec3 dim, string sceneFilename, float tau, ParticleSystem *particleSystem, dim3 blockDim);
	virtual ~LBM3D_1D_indices();

	virtual void recalculateVariables();


	virtual void initScene();

	virtual void draw(ShaderProgram &shader);

	virtual void doStep();
	virtual void doStepCUDA();
	virtual void clearBackLattice();
	virtual void streamingStep();
	virtual void collisionStep();
	virtual void moveParticles();
	virtual void updateInlets();
	virtual void updateColliders();

	virtual void resetSimulation();

	virtual void updateControlProperty(eLBMControlProperty controlProperty);

	virtual void switchToCPU();

protected:

	virtual void swapLattices();
	virtual void initBuffers();
	virtual void initLattice();

private:

	int frameId = 0; ///< Frame id for debugging

	int respawnY = 0;	///< Respawn y coordinate
	int respawnZ = 0;	///< Respawn z coordinate

	/// Returns flattened index.
	int getIdx(int x, int y, int z) {
		return (x + latticeWidth * (y + latticeHeight * z));
	}

	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	dim3 blockDim;		///< Dimension of the CUDA blocks
	dim3 gridDim;		///< Dimension of the CUDA grid
	int cacheSize;		///< Size of the shared memory cache (in Bytes!!!)


	/// Calculates macroscopic density of the node at coordinates x, y, z.
	float calculateMacroscopicDensity(int x, int y, int z);

	/// Calculates macroscopic velocity of the node at coordinates x, y, z with the specified macroscopic density.
	glm::vec3 calculateMacroscopicVelocity(int x, int y, int z, float macroDensity);


};

