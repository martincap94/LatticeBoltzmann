#pragma once

#include "Config.h"
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>


const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f);

enum CameraMovementDirection {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN,
	ROTATE_LEFT,
	ROTATE_RIGHT
};

enum CameraView {
	VIEW_FRONT,
	VIEW_SIDE,
	VIEW_TOP
};

class Camera {
public:

	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;

	float yaw;
	float pitch;
	float roll;

	float movementSpeed = CAMERA_VELOCITY;

	Camera();
	Camera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	~Camera();

	glm::mat4 getViewMatrix();
	void processKeyboardMovement(CameraMovementDirection direction, float deltaTime);
	void setView(CameraView camView);
	void ProcessMouseScroll(float yoffset);
	void printInfo();

private:

	void updateCameraVectors();

};

