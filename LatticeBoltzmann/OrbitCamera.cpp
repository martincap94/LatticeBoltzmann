#include "OrbitCamera.h"

#include <iostream>
#include "glm/gtx/string_cast.hpp"

OrbitCamera::OrbitCamera() {
}

OrbitCamera::OrbitCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch, glm::vec3 focusPoint) : Camera(position, up, yaw, pitch), focusPoint(focusPoint) {
	updateCameraVectors();
}



OrbitCamera::~OrbitCamera() {
}

glm::mat4 OrbitCamera::getViewMatrix() {
	return glm::lookAt(position, position + front, up);
}


void OrbitCamera::processKeyboardMovement(CameraMovementDirection direction, float deltaTime) {
	float velocity = movementSpeed * deltaTime;

	if (direction == FORWARD) {
		position += front * velocity;
		focusPoint += front * velocity;

	}
	if (direction == BACKWARD) {
		position -= front * velocity;
		focusPoint -= front * velocity;

	}
	if (direction == LEFT) {
		position -= right * velocity;
		focusPoint -= right * velocity;

	}
	if (direction == RIGHT) {
		position += right * velocity;
		focusPoint += right * velocity;

	}
	if (direction == UP) {
		position += up * velocity;
		focusPoint += up * velocity;

	}
	if (direction == DOWN) {
		position -= up * velocity;
		focusPoint -= up * velocity;

	}
	if (direction == ROTATE_LEFT) {
		yaw -= velocity;
		updateCameraVectors();
	}
	if (direction == ROTATE_RIGHT) {
		yaw += velocity;
		updateCameraVectors();
	}
	/*if (direction == ROTATE_LEFT) {
	yaw -= velocity;
	updateCameraVectors();
	}
	if (direction == ROTATE_RIGHT) {
	yaw += velocity;
	updateCameraVectors();
	}*/
}

void OrbitCamera::ProcessMouseScroll(float yoffset) {

	// TO DO?
}


void OrbitCamera::setView(CameraView camView) {
	switch (camView) {
		case VIEW_FRONT:
			position = glm::vec3(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, GRID_DEPTH * 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_SIDE:
			position = glm::vec3(GRID_WIDTH * 2.0f, GRID_HEIGHT / 2.0f, GRID_DEPTH / 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(0.0f, 0.0f, -1.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		case VIEW_TOP:
			position = glm::vec3(GRID_WIDTH / 2.0f, GRID_HEIGHT * 2.0f, GRID_DEPTH / 2.0f);
			front = glm::normalize(initFocusPoint - position);
			right = glm::vec3(1.0f, 0.0f, 0.0f);
			up = glm::normalize(glm::cross(right, front));
			break;
		
	}
}

void OrbitCamera::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}

void OrbitCamera::updateCameraVectors() {
	//float x = radius * sin(glm::radians(pitch)) * cos(glm::radians(yaw));
	//float y = radius * cos(glm::radians(pitch));
	//float z = radius * sin(glm::radians(pitch)) * sin(glm::radians(yaw));

	float x = radius * cos(glm::radians(yaw));
	float y = pitch;
	float z = radius * sin(glm::radians(yaw));
	position = focusPoint + glm::vec3(x, y, z);

	front = glm::normalize(focusPoint - position);
	right = glm::normalize(glm::cross(front, WORLD_UP));
	up = glm::normalize(glm::cross(right, front));

	//glm::vec3 tmp;
	//tmp.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	//tmp.y = sin(glm::radians(pitch));
	//tmp.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	//front = glm::normalize(tmp);
	//right = glm::normalize(glm::cross(front, WORLD_UP));
	//up = glm::normalize(glm::cross(right, front));
}