#pragma once

#include <glad\glad.h>

#include "Config.h"
#include "ShaderProgram.h"


class GeneralGrid {
public:
	GeneralGrid();
	GeneralGrid(int range, int stepSize);
	~GeneralGrid();

	void draw(ShaderProgram &shader);

private:

	GLuint VAO;
	GLuint VBO;

	int numLines;

	int range;
	int stepSize;


};

