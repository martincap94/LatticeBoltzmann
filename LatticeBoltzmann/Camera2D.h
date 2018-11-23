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

	virtual glm::mat4 getViewMatrix();
	virtual void processKeyboardMovement(CameraMovementDirection direction, float deltaTime);
	virtual void setView(CameraView camView);
	virtual void processMouseScroll(float yoffset);
	virtual void printInfo();

private:

	virtual void updateCameraVectors();

};

