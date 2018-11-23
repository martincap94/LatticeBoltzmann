#pragma once

#include <string>
#include <glad\glad.h>

#include "ShaderProgram.h"

#include "Config.h"


class HeightMap {
public:

	int width;
	int height;

	int maxIntensity;

	float **data;


	HeightMap();
	HeightMap(string filename, ShaderProgram *shader);
	~HeightMap();

	void draw();

private:

	GLuint VAO;
	GLuint VBO;

	ShaderProgram *shader;

	int numPoints = 0;



};

