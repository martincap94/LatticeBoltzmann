#pragma once

#include "ShaderProgram.h"

class LBM {
public:

	int *d_numParticles;

	int latticeWidth;
	int latticeHeight;
	int latticeDepth;

	int latticeSize;

	float tau = 0.52f;
	float itau;
	float nu;

	string sceneFilename;

	LBM();
	LBM(glm::vec3 dimensions, string sceneFilename, float tau);
	~LBM();

	virtual void initScene() = 0;

	virtual void draw(ShaderProgram &shader) = 0;
	virtual void doStep() = 0;
	virtual void doStepCUDA() = 0;
	virtual void clearBackLattice() = 0;
	virtual void streamingStep() = 0;
	virtual void collisionStep() = 0;
	virtual void moveParticles() = 0;
	virtual void updateInlets() = 0;
	virtual void updateColliders() = 0;
	virtual void resetSimulation() = 0;


};

