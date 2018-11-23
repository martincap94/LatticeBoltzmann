#pragma once

#include <glm\glm.hpp>
#include "ShaderProgram.h"

class Particles {
public:

	int numParticles;
	glm::vec3 *particleVertices;

	Particles();
	Particles(int numParticles);
	~Particles();

	void draw(const ShaderProgram &shader);

private:

	GLuint vbo;
	GLuint vao;

};

