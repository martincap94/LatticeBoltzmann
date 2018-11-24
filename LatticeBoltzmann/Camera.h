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

	int latticeWidth;
	int latticeHeight;
	int latticeDepth;

	Camera();
	Camera(glm::vec3 position, glm::vec3 up = WORLD_UP, float yaw = -90.0f, float pitch = 0.0f);
	~Camera();

	virtual glm::mat4 getViewMatrix() = 0;
	virtual void processKeyboardMovement(CameraMovementDirection direction, float deltaTime) = 0;
	virtual void processMouseScroll(float yoffset) = 0;
	virtual void setView(CameraView camView);
	virtual void printInfo();

	void setLatticeDimensions(int latticeWidth, int latticeHeight, int latticeDepth);

protected:

	virtual void updateCameraVectors() = 0;

};

