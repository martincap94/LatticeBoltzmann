#pragma once

#include <glm\glm.hpp>

const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f);

class Camera {

public:

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


	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;

	float yaw;
	float pitch;
	float roll;

	Camera();
	Camera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	~Camera();

protected:

	virtual void updateCameraVectors() = 0;

};

