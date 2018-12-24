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


	Node3D *frontLattice;
	Node3D *backLattice;

	Node3D *d_frontLattice;
	Node3D *d_backLattice;

	ParticleSystem *particleSystem;
	glm::vec3 *particleVertices;
	glm::vec3 *d_particleVertices;

	glm::vec3 *velocities;

	glm::vec3 *d_velocities;

	bool *testCollider;
	bool *d_testCollider;
	vector<glm::vec3> colliderVertices;

	HeightMap *heightMap;
	float *d_heightMap;


	struct cudaGraphicsResource *cuda_vbo_resource;



	LBM3D_1D_indices();
	LBM3D_1D_indices(glm::vec3 latticeDim, string sceneFilename, float tau, ParticleSystem *particleSystem, dim3 blockDim);
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

private:

	int frameId = 0; // for debugging

	int respawnY = 0;
	int respawnZ = 0;

	int getIdx(int x, int y, int z) {
		return (x + latticeWidth * (y + latticeHeight * z));
	}

	GLuint velocityVBO;
	GLuint velocityVAO;

	GLuint particleArrowsVAO;
	GLuint particleArrowsVBO;

	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	dim3 blockDim;
	dim3 gridDim;
	int sharedMemCacheSize;

	void initBuffers();
	void initLattice();

	float calculateMacroscopicDensity(int x, int y, int z);
	glm::vec3 calculateMacroscopicVelocity(int x, int y, int z, float macroDensity);


};

