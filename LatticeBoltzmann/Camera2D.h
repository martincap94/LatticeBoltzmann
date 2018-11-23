#pragma once

#include "Config.h"
#include "Camera.h"

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>


class Camera2D : public Camera {
public:

	float movementSpeed = CAMERA_VELOCITY;

	Camera2D();
	Camera2D(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	~Camera2D();

	glm::mat4 getViewMatrix();
	void processKeyboardMovement(CameraMovementDirection direction, float deltaTime);
	void setView(CameraView camView);
	void ProcessMouseScroll(float yoffset);
	void printInfo();

private:

	virtual void updateCameraVectors();

};

