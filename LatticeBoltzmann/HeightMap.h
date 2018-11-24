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

	ShaderProgram *shader;



	HeightMap();
	HeightMap(string filename, int latticeHeight, ShaderProgram *shader);
	~HeightMap();

	void draw();

private:

	GLuint VAO;
	GLuint VBO;


	int numPoints = 0;



};

