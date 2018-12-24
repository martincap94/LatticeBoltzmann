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
/// Lattice node for 2D simulation (9 streaming directions -> 9 floats in distribution function).
struct Node {
	float adj[9];
};

// Streaming directions vectors
const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

/// Streaming directions array.
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


/// Streaming direction enum for 2D.
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

/// 2D LBM simulator.
/**
	2D LBM simulator that supports both CPU and GPU simulations.
	GPU (CUDA) simulation is run through global kernels that are defined in LBM2D_1D_indices.cu.
	The LBM is indexed as a 1D array in this implementation.
	The simulator supports particle velocity visualization. Stream line and velocity vector
	visualizations have been deprecated.
*/
class LBM2D_1D_indices : public LBM {

	const float WEIGHT_MIDDLE = 4.0f / 9.0f;		///< Initial weight for the middle value in distribution function
	const float WEIGHT_AXIS = 1.0f / 9.0f;			///< Initial weight for all values in distribution function that lie on the axes
	const float WEIGHT_DIAGONAL = 1.0f / 36.0f;		///< Initial weight for all values in distribution function that lie on the diagonal


public:

	Node *frontLattice;			///< Front lattice - the one currently drawn at end of each frame
	Node *backLattice;			///< Back lattice - the one to which we prepare next frame to be drawn

	Node *d_frontLattice;		///< Device pointer for the front lattice
	Node *d_backLattice;		///< Device pointer for the back lattice
	glm::vec2 *d_velocities;	///< Device pointer to the velocities array

	bool *d_tCol;				///< Device pointer to the scene collider (scene descriptor)



	ParticleSystem *particleSystem;		///< Pointer to the particle system
	glm::vec3 *particleVertices;		///< Pointer to the particle vertices array (on CPU)


	LatticeCollider *tCol;				///< Scene collider (scene descriptor)

	struct cudaGraphicsResource *cudaParticleVerticesVBO;	///< Device pointer that is mapped to particle vertices VBO
	struct cudaGraphicsResource *cudaParticleColorsVBO;		///< Device pointer that is mapped to particle colors VBO


	glm::vec2 *velocities;				///< Macroscopic velocities array
	vector<glm::vec3> velocityArrows;	///< Array describing velocity arrows (starts in node, points in velocity direction) for visualization
	vector<glm::vec3> particleArrows;	///< Array describing velocity arrows that visualize particle velocity (interpolated values)


	/// Default constructor.
	LBM2D_1D_indices();

	/// Constructs the simulator with given dimensions, scene, initial tau value and number of threads for launching kernels.
	/**
		Constructs the simulator with given dimensions, scene, initial tau value and number of threads for launching kernels.
		Initializes the scene and allocates CPU and GPU memory for simulation.
	*/
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

