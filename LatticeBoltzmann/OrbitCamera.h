#pragma once

#include "Config.h"
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include "Camera.h"


class OrbitCamera {
public:

	glm::vec3 position;
	glm::vec3 focusPoint;
	glm::vec3 initFocusPoint;

	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;

	float yaw;
	float pitch;
	float roll;

	float radius = 100.0f;

	float movementSpeed = CAMERA_VELOCITY;

	OrbitCamera();
	OrbitCamera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f, glm::vec3 focusPoint = glm::vec3(0.0f));
	~OrbitCamera();

	glm::mat4 getViewMatrix();
	void processKeyboardMovement(CameraMovementDirection direction, float deltaTime);
	void setView(CameraView camView);
	void ProcessMouseScroll(float yoffset);
	void printInfo();

private:

	void updateCameraVectors();

};

