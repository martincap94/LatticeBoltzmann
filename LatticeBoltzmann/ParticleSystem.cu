#include "ParticleSystem.h"

#include <cuda_runtime.h>

#define PARTICLES_3D

ParticleSystem::ParticleSystem() {
}

ParticleSystem::ParticleSystem(int numParticles, bool drawStreamlines) : numParticles(numParticles), drawStreamlines(drawStreamlines) {
	particleVertices = new glm::vec3[numParticles]();



	cudaMalloc((void**)&d_numParticles, sizeof(int));

	cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice);


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


	if (drawStreamlines) {
		streamLines = new glm::vec3[numParticles * MAX_STREAMLINE_LENGTH];

		glGenVertexArrays(1, &streamLinesVAO);
		glBindVertexArray(streamLinesVAO);
		glGenBuffers(1, &streamLinesVBO);
		glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

		glBindVertexArray(0);
	}



}


ParticleSystem::~ParticleSystem() {
	delete[] particleVertices;

	if (streamLines != nullptr) {
		delete[] streamLines;
	}
	//cudaFree(d_numParticles);
}

void ParticleSystem::draw(const ShaderProgram &shader, bool useCUDA) {

	glUseProgram(shader.id);

	glPointSize(2.0f);
	shader.setVec4("color", particlesColor);

	glBindVertexArray(vao);

	if (!useCUDA) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
	}

	glDrawArrays(GL_POINTS, 0, numParticles);

	if (drawStreamlines) {

		glPointSize(1.0f);
		shader.setVec4("color", glm::vec4(0.0f, 0.4f, 1.0f, 1.0f));

		glBindVertexArray(streamLinesVAO);

		glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles * MAX_STREAMLINE_LENGTH, &streamLines[0], GL_DYNAMIC_DRAW);

		glDrawArrays(GL_POINTS, 0, numParticles  * MAX_STREAMLINE_LENGTH);
	}


}

void ParticleSystem::initParticlePositions(int width, int height) {
	int particleCount = 0;
	int x = 0;
	int y = 0;
	float offset = 0.1f;
	while (particleCount != numParticles) {
		particleVertices[particleCount] = glm::vec3(x, y++, -1.0f);
		if (y >= height - 1) {
			y = 0;
			x++;
		}
		if (x >= width - 1) {
			x = offset;
			y = offset;
			offset += 0.1f;
		}
		particleCount++;
	}
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
}

void ParticleSystem::initParticlePositions(int width, int height, int depth) {


	// generate in the left wall
	int particleCount = 0;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	while (particleCount != numParticles) {
		// prefer depth instead of height
		if (z >= depth - 1) {
			z = 0.0f;
			y++;
		}
		if (y >= height - 1) {
			y = 0.0f;
			z = 0.0f;
			x++;
		}
		if (x >= width - 1) {
			x = 0.5f;
			y = 0.5f;
			z = 0.5f;
		}
		particleVertices[particleCount] = glm::vec3(x, y, z++);

		particleCount++;
	}
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
}
