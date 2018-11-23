#pragma once

#include "Config.h"
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

#include "Camera2D.h"
#include "Camera.h"

class OrbitCamera : public Camera{
public:

	glm::vec3 focusPoint;
	glm::vec3 initFocusPoint;


	float radius = 100.0f;

	float movementSpeed = CAMERA_VELOCITY;

	OrbitCamera();
	OrbitCamera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f, glm::vec3 focusPoint = glm::vec3(0.0f));
	~OrbitCamera();

	virtual glm::mat4 getViewMatrix();
	virtual void processKeyboardMovement(CameraMovementDirection direction, float deltaTime);
	virtual void setView(CameraView camView);
	virtual void processMouseScroll(float yoffset);
	virtual void printInfo();

private:

	virtual void updateCameraVectors();

};

