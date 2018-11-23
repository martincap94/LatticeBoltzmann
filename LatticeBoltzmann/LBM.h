#pragma once

#include "ShaderProgram.h"

class LBM {
public:

	LBM();
	~LBM();

	virtual void draw(ShaderProgram &shader) = 0;
	virtual void doStep() = 0;
	virtual void doStepCUDA() = 0;
	virtual void clearBackLattice() = 0;
	virtual void streamingStep() = 0;
	virtual void collisionStep() = 0;
	virtual void moveParticles() = 0;
	virtual void updateInlets() = 0;
	virtual void updateColliders() = 0;


};

