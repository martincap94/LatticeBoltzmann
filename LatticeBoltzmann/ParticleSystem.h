#pragma once

#include <glm\glm.hpp>
#include "ShaderProgram.h"
#include "HeightMap.h"
#include <vector>
#include <deque>

#include "Texture.h"

class ParticleSystem {
public:

	int numParticles;
	int *d_numParticles;
	int maxNumParticles = 10000;

	bool drawStreamlines = false;

	int pointSize = 2;

	glm::vec3 particlesColor = glm::vec3(0.8f, 0.8f, 0.8f);

	Texture spriteTexture;

	glm::vec3 *particleVertices = nullptr;
	glm::vec3 *streamLines = nullptr;

	ParticleSystem();
	ParticleSystem(int numParticles, bool drawStreamlines);
	~ParticleSystem();

	void draw(const ShaderProgram &shader, bool useCUDA);
	void initParticlePositions(int width, int height, bool *collider);
	void initParticlePositions(int width, int height, int depth);


	GLuint vbo;

private:

	GLuint vao;

	GLuint streamLinesVAO;
	GLuint streamLinesVBO;

};

