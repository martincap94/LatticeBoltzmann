#include "Camera.h"

#include <iostream>
#include "glm/gtx/string_cast.hpp"

Camera::Camera() {
}

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
	: position(position), up(up), yaw(yaw), pitch(pitch) {
	updateCameraVectors();
}



Camera::~Camera() {
}

glm::mat4 Camera::getViewMatrix() {
	return glm::lookAt(position, position + front, up);
}


void Camera::processKeyboardMovement(CameraMovementDirection direction, float deltaTime) {
	float velocity = movementSpeed * deltaTime;

	if (direction == FORWARD) {
		position += front * velocity;
	}
	if (direction == BACKWARD) {
		position -= front * velocity;
	}
	if (direction == LEFT) {
		position -= right * velocity;
	}
	if (direction == RIGHT) {
		position += right * velocity;
	}
	if (direction == UP) {
		position += up * velocity;
	}
	if (direction == DOWN) {
		position -= up * velocity;
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

void Camera::setView(CameraView camView) {
	switch (camView) {
		case VIEW_TOP:

			break;

	}
}

void Camera::ProcessMouseScroll(float yoffset) {
	/*if (Zoom >= 1.0f && Zoom <= 45.0f) {
		Zoom -= yoffset;
	}
	if (Zoom <= 1.0f) {
		Zoom = 1.0f;
	}
	if (Zoom >= 45.0f) {
		Zoom = 45.0f;
	}*/
}

void Camera::printInfo() {
	cout << "Camera position: " << glm::to_string(position) << endl;
}

void Camera::updateCameraVectors() {
	glm::vec3 tmp;
	tmp.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	tmp.y = sin(glm::radians(pitch));
	tmp.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	front = glm::normalize(tmp);
	right = glm::normalize(glm::cross(front, WORLD_UP));
	up = glm::normalize(glm::cross(right, front));
}
