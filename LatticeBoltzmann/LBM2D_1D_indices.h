#pragma once

#include "Config.h"

#include "ShaderProgram.h"
#include "ParticleSystem.h"
#include "LatticeCollider.h"

#include <cuda_gl_interop.h>


#include <vector>

#include "LBM.h"

// temporary -> will be moved to special header file to be shared
// among all classes (Node -> Node2D and Node3D)
// this applies to Node, vRight, ..., EDirection
struct Node {
	float adj[9];
};


const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

const glm::vec3 directionVectors[9] = {
	glm::vec3(0.0f, 0.0f, 0.0f),
	glm::vec3(1.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, -1.0f, 0.0f),
	glm::vec3(1.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, -1.0f, 0.0f),
	glm::vec3(1.0f, -1.0f, 0.0f)
};



enum EDirection {
	DIR_MIDDLE = 0,
	DIR_RIGHT,
	DIR_TOP,
	DIR_LEFT,
	DIR_BOTTOM,
	DIR_TOP_RIGHT,
	DIR_TOP_LEFT,
	DIR_BOTTOM_LEFT,
	DIR_BOTTOM_RIGHT,
	NUM_2D_DIRECTIONS
};


class LBM2D_1D_indices : public LBM {

	const float WEIGHT_MIDDLE = 4.0f / 9.0f;
	const float WEIGHT_AXIS = 1.0f / 9.0f;
	const float WEIGHT_DIAGONAL = 1.0f / 36.0f;


public:

	Node *frontLattice;
	Node *backLattice;

	Node *d_frontLattice;
	Node *d_backLattice;
	glm::vec2 *d_velocities;

	bool *d_tCol;



	ParticleSystem *particleSystem;
	glm::vec3 *particleVertices;
	glm::vec3 *d_particleVertices;


	LatticeCollider *tCol;

	struct cudaGraphicsResource *cudaParticleVerticesVBO;
	struct cudaGraphicsResource *cudaParticleColorsVBO;


	glm::vec2 *velocities;
	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	LBM2D_1D_indices();
	LBM2D_1D_indices(glm::vec3 dim, string sceneFilename, float tau, ParticleSystem *particleSystem, int numThreads);
	virtual ~LBM2D_1D_indices();


	virtual void recalculateVariables();

	virtual void initScene();

	virtual void draw(ShaderProgram &shader);

	virtual void doStep();
	virtual void doStepCUDA();

	virtual void clearBackLattice();
	virtual void streamingStep();
	virtual void collisionStep();
	void collisionStepStreamlined();

	virtual void moveParticles();
	virtual void updateInlets();
	void updateInlets(Node *lattice);
	virtual void updateColliders();

	virtual void resetSimulation();

	virtual void updateControlProperty(eLBMControlProperty controlProperty);

	virtual void switchToCPU();

protected:
	virtual void swapLattices();


private:

	int numThreads;
	int numBlocks;

	GLuint vbo;
	GLuint vao;

	GLuint velocityVBO;
	GLuint velocityVAO;

	GLuint particleArrowsVAO;
	GLuint particleArrowsVBO;

	int respawnIndex = 0;
	int respawnMinY;
	int respawnMaxY;

	int streamLineCounter = 0;

	void initBuffers();

	void initLattice();
	void precomputeRespawnRange();

	int getIdx(int x, int y) {
		return x + y * latticeWidth;
	}


	float calculateMacroscopicDensity(int x, int y);
	glm::vec3 calculateMacroscopicVelocity(int x, int y, float macroDensity);



};

