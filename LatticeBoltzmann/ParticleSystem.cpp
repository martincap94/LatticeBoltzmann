#include "ParticleSystem.h"

#include <cuda_runtime.h>

#define PARTICLES_3D

ParticleSystem::ParticleSystem() {
}

ParticleSystem::ParticleSystem(int numParticles) : numParticles(numParticles) {
	particleVertices = new glm::vec3[numParticles];
	//streamLines.reserve(numParticles);
	//for (int i = 0; i < numParticles; i++) {
	//	//streamLines[i] = new glm::vec3[MAX_STREAMLINE_LENGTH]();
	//	//streamLines[i].re(MAX_STREAMLINE_LENGTH);
	//	streamLines.push_back(deque<glm::vec3>());
	//}
	streamLines = new glm::vec3[numParticles * MAX_STREAMLINE_LENGTH];

	cudaMalloc((void**)&d_numParticles, sizeof(int));

	cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice);

	//float y = 0.5f;
	//for (int i = 0; i < numParticles; i++) {
	//	//particleVertices[i] = glm::vec3(rand() / (RAND_MAX / 4.0f), y++, -1.0f);
	//	particleVertices[i] = glm::vec3(4.0f, y++, -1.0f);
	//	//particleVertices[i] = glm::vec3(rand() / (RAND_MAX / (GRID_WIDTH - 1)),
	//	//								rand() / (RAND_MAX / (GRID_HEIGHT - 1)),
	//	//								-1.0f);
	//}
#ifdef RUN_LBM3D

	// generate in the left wall
	int particleCount = 0;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	while (particleCount != numParticles) {
		particleVertices[particleCount] = glm::vec3(x, y, z++);
		// prefer depth instead of height
		if (z >= GRID_DEPTH - 1) {
			z = 0.0f;
			y++;
		}
		if (y >= GRID_HEIGHT - 1) {
			y = 0.0f;
			z = 0.0f;
			x++;
		}
		if (x >= GRID_WIDTH - 1) {
			x = 0.5f;
			y = 0.5f;
			z = 0.5f;
		}
		particleCount++;
	}

#endif
#ifdef RUN_LBM2D

	int particleCount = 0;
	int x = 0;
	int y = 0;
	float offset = 0.1f;
	while (particleCount != numParticles) {
		particleVertices[particleCount] = glm::vec3(x, y++, -1.0f);
		if (y >= GRID_HEIGHT - 1) {
			y = 0;
			x++;
		}
		if (x >= GRID_WIDTH - 1) {
			x = offset;
			y = offset;
			offset += 0.1f;
		}
		particleCount++;
	}
#endif

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);

	glGenVertexArrays(1, &streamLinesVAO);
	glBindVertexArray(streamLinesVAO);
	glGenBuffers(1, &streamLinesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);



}


ParticleSystem::~ParticleSystem() {
	delete[] particleVertices;
	delete[] streamLines;
}

void ParticleSystem::draw(const ShaderProgram & shader) {

	glUseProgram(shader.id);

	glPointSize(2.0f);
	shader.setVec4("color", glm::vec4(1.0f, 0.4f, 1.0f, 1.0f));

	glBindVertexArray(vao);

#ifndef USE_INTEROP
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
#endif

	glDrawArrays(GL_POINTS, 0, numParticles);

#ifndef USE_CUDA

	glPointSize(1.0f);
	shader.setVec4("color", glm::vec4(0.0f, 0.4f, 1.0f, 1.0f));

	glBindVertexArray(streamLinesVAO);

	glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles * MAX_STREAMLINE_LENGTH, &streamLines[0], GL_DYNAMIC_DRAW);

	glDrawArrays(GL_POINTS, 0, numParticles  * MAX_STREAMLINE_LENGTH);
#endif


}

void ParticleSystem::initParticlePositions(HeightMap *heightMap) {

	//particleVertices[0] = glm::vec3(-10.0f, -10.0f, -10.0f);

	//int idx = 0;
	//for (float z = 0.0f; z < GRID_DEPTH; z += 1.0f) {
	//	for (float y = 0.0f; y < GRID_HEIGHT; y += 1.0f) {
	//		if (y > heightMap->data[0][(int)z]) {
	//			particleVertices[idx++] = glm::vec3(0.0f, y, z);
	//			if (idx >= numParticles) {
	//				break;
	//			}
	//		}
	//	}
	//}

}
