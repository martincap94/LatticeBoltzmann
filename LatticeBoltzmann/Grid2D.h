#pragma once

#include "Config.h"

#include <glm\glm.hpp>

#include <glad/glad.h>
#include <vector>
#include "ShaderProgram.h"

#include "Grid.h"

class Grid2D : public Grid {

public:

	GLuint vao;
	GLuint vbo;

	//glm::vec2 *gridVertices;
	vector<glm::vec3> gridVertices;

	Grid2D(int width, int height, int stepX = 1, int stepY = 1);
	~Grid2D();

	virtual void draw(ShaderProgram &shader);
};

