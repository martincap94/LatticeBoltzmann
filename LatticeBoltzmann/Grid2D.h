#pragma once

#include "Config.h"

#include <glm\glm.hpp>

#include <glad/glad.h>
#include <vector>
#include "ShaderProgram.h"


class Grid2D {

public:

	GLuint vao;
	GLuint vbo;

	//glm::vec2 *gridVertices;
	vector<glm::vec3> gridVertices;

	Grid2D();
	~Grid2D();

	void draw(ShaderProgram &shader);
};

