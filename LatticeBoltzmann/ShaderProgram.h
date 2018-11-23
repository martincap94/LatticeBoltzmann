#pragma once

#include <glad\glad.h>
#include <GLFW\glfw3.h>

#include <string>
#include "Config.h"

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>



class ShaderProgram {
public:

	GLuint id;

	ShaderProgram();
	ShaderProgram(const GLchar *vsPath, const GLchar *fsPath);
	~ShaderProgram();

	void setMat4fv(const string &name, glm::mat4 value) const;
	void setVec3(const std::string &name, float x, float y, float z) const {
		glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
	}

	void setVec3(const std::string &name, glm::vec3 value) const {
		glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
	}

	void setVec4(const std::string &name, glm::vec4 value) const {
		glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
	}
};

