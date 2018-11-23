#pragma once

#include "Config.h"
#include <string>
#include "ShaderProgram.h"

class LatticeCollider {
public:

	int width;
	int height;
	int maxIntensity;

	bool *area;

	//LatticeCollider();
	LatticeCollider(string filename);
	~LatticeCollider();

	void draw(ShaderProgram &shader);

private:

	int numPoints = 0;
	GLuint VAO;
	GLuint VBO;

};

