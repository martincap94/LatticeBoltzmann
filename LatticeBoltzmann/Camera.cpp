#include "Camera.h"



Camera::Camera() {
}

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
	: position(position), up(up), yaw(yaw), pitch(pitch) {
}



Camera::~Camera() {
}


//void Camera::updateCameraVectors() {
//}
