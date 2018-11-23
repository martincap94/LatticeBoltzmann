#pragma once

#include "Config.h"

#include <glm\glm.hpp>

#include <glad\glad.h>
#include <vector>
#include "ShaderProgram.h"


class Grid3D {
public:
	Grid3D(int stepX = 1, int stepY = 1, int stepZ = 1);
	~Grid3D();

	void draw(ShaderProgram &shader);

private:

	GLuint VAO;
	GLuint VBO;

	GLuint pointsVAO;
	GLuint pointsVBO;

	GLuint boxVAO;
	GLuint boxVBO;


	vector<glm::vec3> gridVertices;


};

