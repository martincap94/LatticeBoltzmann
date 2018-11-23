#pragma once

#include <glm\glm.hpp>
#include "ShaderProgram.h"
#include "HeightMap.h"
#include <vector>
#include <deque>

class ParticleSystem {
public:

	int numParticles;
	glm::vec3 *particleVertices;
	//vector<glm::vec3> particleVerticesVector;
	//vector<deque<glm::vec3>> streamLines;
	glm::vec3 *streamLines;

	ParticleSystem();
	ParticleSystem(int numParticles);
	~ParticleSystem();

	void draw(const ShaderProgram &shader);
	void initParticlePositions(HeightMap *heightMap);

	GLuint vbo;

private:

	//GLuint vbo;
	GLuint vao;

	GLuint streamLinesVAO;
	GLuint streamLinesVBO;

};

