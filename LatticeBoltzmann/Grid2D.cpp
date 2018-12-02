#include "Grid2D.h"


Grid2D::Grid2D() {
	//gridVertices = new glm::vec2[GRID_WIDTH * GRID_HEIGHT];

	//int idx = 0;
	for (int x = 0; x < GRID_WIDTH; x++) {
		//gridVertices[idx++] = glm::vec2(x, 0.0f);
		//gridVertices[idx++] = glm::vec2(x, GRID_HEIGHT);
		gridVertices.push_back(glm::vec3(x, 0.0f, -2.0f));
		gridVertices.push_back(glm::vec3(x, GRID_HEIGHT - 1, -2.0f));
	}
	for (int y = 0; y < GRID_HEIGHT; y++) {
		//gridVertices[idx++] = glm::vec2(0.0f, y);
		//gridVertices[idx++] = glm::vec2(GRID_WIDTH, y);
		gridVertices.push_back(glm::vec3(0.0f, y, -2.0f));
		gridVertices.push_back(glm::vec3(GRID_WIDTH - 1, y, -2.0f));
	}

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(glm::vec3), &gridVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
	
	glBindVertexArray(0);


}


Grid2D::~Grid2D() {
	//delete[] gridVertices;
}

void Grid2D::draw(ShaderProgram &shader) {
	glUseProgram(shader.id);
	shader.setVec3("uColor", glm::vec3(0.1f, 0.1f, 0.1f));
	glBindVertexArray(vao);
	glDrawArrays(GL_LINES, 0, gridVertices.size());


}
