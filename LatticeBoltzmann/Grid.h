#pragma once

#include "ShaderProgram.h"

class Grid {
public:
	Grid();
	~Grid();

	virtual void draw(ShaderProgram &shader) = 0;
};

