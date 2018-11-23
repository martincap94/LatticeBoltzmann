#pragma once

#include "Config.h"

#include "ShaderProgram.h"
#include "ParticleSystem.h"
#include "LatticeCollider.h"

#include <cuda_gl_interop.h>


#include <vector>

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

enum EDirection {
	DIR_MIDDLE = 0,
	DIR_RIGHT,
	DIR_TOP,
	DIR_LEFT,
	DIR_BOTTOM,
	DIR_TOP_RIGHT,
	DIR_TOP_LEFT,
	DIR_BOTTOM_LEFT,
	DIR_BOTTOM_RIGHT
};


class LBM2D_1D_indices {
	
	/*struct Node {
		float adj[9];
	};*/

	//struct Node {
	//	float adj[9];
	//};

	


	//const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	//const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	//const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	//const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	//const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	//const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	//const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
	//const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

	//enum EDirection {
	//	DIR_MIDDLE = 0,
	//	DIR_RIGHT,
	//	DIR_TOP,
	//	DIR_LEFT,
	//	DIR_BOTTOM,
	//	DIR_TOP_RIGHT,
	//	DIR_TOP_LEFT,
	//	DIR_BOTTOM_LEFT,
	//	DIR_BOTTOM_RIGHT
	//};

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

	//bool **testCollider;
	//glm::vec3 testColliderExtent[2];

	LatticeCollider *tCol;

	struct cudaGraphicsResource *cuda_vbo_resource;

	//GLuint tcVAO;
	//GLuint tcVBO;

	glm::vec2 *velocities;
	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	LBM2D_1D_indices();
	LBM2D_1D_indices(ParticleSystem *particleSystem);
	~LBM2D_1D_indices();

	void draw(ShaderProgram &shader);

	void doStep();
	void doStepCUDA();

	void clearBackLattice();
	void streamingStep();
	void collisionStep();
	void collisionStepStreamlined();
	void collisionStepCUDA();

	void moveParticles();
	void updateInlets();
	void updateInlets(Node *lattice);
	void updateColliders();
	void updateCollidersAlt();


private:

	GLuint vbo;
	GLuint vao;

	GLuint velocityVBO;
	GLuint velocityVAO;

	GLuint particleArrowsVAO;
	GLuint particleArrowsVBO;

	int respawnIndex = 0;

	int streamLineCounter = 0;

	void initBuffers();

	void initLattice();
	void initTestCollider();

	int getIdx(int x, int y) {
		return x + y * GRID_WIDTH;
	}


	void swapLattices();
	float calculateMacroscopicDensity(int x, int y);
	glm::vec3 calculateMacroscopicVelocity(int x, int y, float macroDensity);



};

