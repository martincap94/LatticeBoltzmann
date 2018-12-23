#pragma once

#include "ShaderProgram.h"



class LBM {
public:

	enum eLBMControlProperty {
		MIRROR_SIDES_PROP
	};

	int *d_numParticles;	// managed in memory by Particle System class (destructor)

	int latticeWidth;
	int latticeHeight;
	int latticeDepth;

	int latticeSize;

	float tau = 0.52f;
	float itau;
	float nu;

	//float inletVelocity = 0.5f;
	glm::vec3 inletVelocity = glm::vec3(1.0f, 0.0f, 0.0f);


	int useSubgridModel = 0;	// experimental - incorrect

	int mirrorSides = 1;		
	int visualizeVelocity = 0;  // used only in 2D at the time

	int respawnLinearly = 0;	// not used yet

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

	virtual void updateControlProperty(eLBMControlProperty controlProperty) = 0;

	virtual void recalculateVariables();

	virtual void switchToCPU() = 0;

};

