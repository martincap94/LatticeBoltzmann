
#include <iostream>


#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "Config.h"

#define GLM_FORCE_CUDA // for glm CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/gtx/string_cast.hpp"

#include <random>
#include <ctime>

#include "LBM2D.h"
#include "LBM2D_reindexed.h"
#include "LBM2D_1D_indices.h"
#include "LBM3D.h"
#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
#include "Grid2D.h"
#include "Grid3D.h"
#include "GeneralGrid.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "OrbitCamera.h"
#include "ParticleSystem.h"
#include "DirectionalLight.h"




#define WIDTH 1000
#define HEIGHT 1000

int runApp();
void processInput(GLFWwindow* window);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);


#ifdef RUN_LBM3D
//Camera camera(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, -10.0f);
OrbitCamera camera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 10.0f, glm::vec3(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, GRID_DEPTH / 2.0f));

#endif
#ifdef RUN_LBM2D
Camera camera(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
#endif
float deltaTime = 0.0f;
float lastFrameTime;

glm::mat4 view;
glm::mat4 projection;


int screenWidth;
int screenHeight;


int main(int argc, char **argv) {
	srand(time(NULL));

	runApp();


}

int runApp() {

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	glfwWindowHint(GLFW_SAMPLES, 12); // enable MSAA with 4 samples

	GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Lattice Boltzmann", nullptr, nullptr);

	if (!window) {
		cerr << "Failed to create GLFW window" << endl;
		glfwTerminate(); // maybe unnecessary according to the documentation
		return -1;
	}

	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cerr << "Failed to initialize GLAD" << endl;
		return -1;
	}

	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);



	ShaderProgram singleColorShader("singleColor.vert", "singleColor.frag");
	ShaderProgram singleColorShaderAlpha("singleColor.vert", "singleColor_alpha.frag");
	ShaderProgram unlitColorShader("unlitColor.vert", "unlitColor.frag");
	ShaderProgram dirLightOnlyShader("dirLightOnly.vert", "dirLightOnly.frag");

	DirectionalLight dirLight;
	dirLight.direction = glm::vec3(1.0f, 45.0f, 1.0f);
	dirLight.ambient = glm::vec3(0.5f);
	dirLight.diffuse = glm::vec3(1.0f, 0.0f, 0.0f);
	dirLight.specular = glm::vec3(0.0f, 1.0f, 0.0f);

	glUseProgram(dirLightOnlyShader.id);

	dirLightOnlyShader.setVec3("dirLight.direction", dirLight.direction);
	dirLightOnlyShader.setVec3("dirLight.ambient", dirLight.ambient);
	dirLightOnlyShader.setVec3("dirLight.diffuse", dirLight.diffuse);
	dirLightOnlyShader.setVec3("dirLight.specular", dirLight.specular);
	dirLightOnlyShader.setVec3("vViewPos", camera.position);

#ifdef RUN_LBM3D
	Grid3D grid(6, 6, 6);
	HeightMap heightMap(HEIGHTMAP_FILENAME, &dirLightOnlyShader);
#endif
#ifdef RUN_LBM2D
	Grid2D grid;
#endif


	glViewport(0, 0, screenWidth, screenHeight);

	float nearPlane = 0.1f;
	float farPlane = 1000.0f;

	float aspectRatio = screenWidth / screenHeight;

#ifdef RUN_LBM2D
	float projWidth = (GRID_WIDTH > GRID_HEIGHT) ? GRID_WIDTH : GRID_HEIGHT;

	projection = glm::ortho(-1.0f, projWidth, -1.0f, projWidth, nearPlane, farPlane);
#endif
#ifdef RUN_LBM3D
	projection = glm::ortho(-50.0f, 50.0f, -50.0f, 50.0f, nearPlane, farPlane);
#endif

	glUseProgram(singleColorShader.id);
	singleColorShader.setMat4fv("projection", projection);
	glUseProgram(singleColorShaderAlpha.id);
	singleColorShaderAlpha.setMat4fv("projection", projection);
	glUseProgram(unlitColorShader.id);
	unlitColorShader.setMat4fv("uProjection", projection);

	glUseProgram(dirLightOnlyShader.id);
	dirLightOnlyShader.setMat4fv("uProjection", projection);

	GeneralGrid gGrid(100, 5);

	ParticleSystem particles(NUM_PARTICLES);

#ifdef RUN_LBM2D

#ifdef USE_REINDEXED_LBM2D
	//LBM2D_reindexed lbm2D(&particles);
	LBM2D_1D_indices lbm2D(&particles);

#else
	LBM2D lbm2D(GRID_WIDTH, GRID_HEIGHT, &particles);
#endif

#endif
#ifdef RUN_LBM3D

#ifdef USE_REINDEXED_LBM3D
	LBM3D_1D_indices lbm3D(&particles, &heightMap);
#else
	LBM3D lbm3D(&particles);
#endif


#endif

	int frameCounter = 0;


	glEnable(GL_MULTISAMPLE); // enable multisampling (on by default)
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	glEnable(GL_DEPTH_TEST);

	glfwSwapInterval(0); // V-Sync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	while (!glfwWindowShouldClose(window)) {

		//cout << "frame " << frameCounter++ << endl;

		float currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;

		//glClearColor(0.1f, 0.4f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		


		glfwPollEvents();
		processInput(window);


#ifdef RUN_LBM2D
#ifdef USE_CUDA
		lbm2D.doStepCUDA();
#else
		lbm2D.doStep();
#endif
#endif
#ifdef RUN_LBM3D
#ifdef USE_CUDA
		lbm3D.doStepCUDA();
#else
		lbm3D.doStep();
#endif
#endif


		view = camera.getViewMatrix();

		glUseProgram(singleColorShader.id);
		singleColorShader.setMat4fv("view", view);
		//singleColorShader.setMat4fv("projection", projection);
		
		glUseProgram(unlitColorShader.id);
		unlitColorShader.setMat4fv("uView", view);

		glUseProgram(dirLightOnlyShader.id);
		dirLightOnlyShader.setMat4fv("uView", view);
		dirLightOnlyShader.setVec3("vViewPos", camera.position);

		//unlitColorShader.setMat4fv("uProjection", projection);

		glUseProgram(singleColorShaderAlpha.id);
		singleColorShaderAlpha.setMat4fv("view", view);


		grid.draw(singleColorShader);

#ifdef RUN_LBM2D
		lbm2D.draw(singleColorShader);
#endif
#ifdef RUN_LBM3D
		lbm3D.draw(singleColorShader);
#endif



		particles.draw(singleColorShaderAlpha);

		//grid.draw(singleColorShaderAlpha);

		gGrid.draw(unlitColorShader);


		//camera.printInfo();



		glfwSwapBuffers(window);

	}

	glfwTerminate();
	return 0;
}




void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.processKeyboardMovement(UP, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.processKeyboardMovement(DOWN, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.processKeyboardMovement(LEFT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.processKeyboardMovement(RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		camera.processKeyboardMovement(ROTATE_RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera.processKeyboardMovement(ROTATE_LEFT, deltaTime);
	}
#ifdef RUN_LBM3D
	if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
		camera.setView(VIEW_FRONT);
	}
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
		camera.setView(VIEW_SIDE);
	}	
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		camera.setView(VIEW_TOP);
	}

#endif
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.ProcessMouseScroll(yoffset);
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		cout << "Cursor Position at (" << xpos << " : " << ypos << ")" << endl;

		//X_ndc = X_screen * 2.0 / VP_sizeX - 1.0;
		//Y_ndc = Y_screen * 2.0 / VP_sizeY - 1.0;
		//Z_ndc = 2.0 * depth - 1.0;
		xpos = xpos * 2.0f / (float)screenWidth - 1.0f;
		ypos = screenHeight - ypos;
		ypos = ypos * 2.0f / (float)screenHeight - 1.0f;

		glm::vec4 mouseCoords(xpos, ypos, 0.0f, 1.0f);
		mouseCoords = glm::inverse(view) * glm::inverse(projection) * mouseCoords;
		cout << "mouse coords = " << glm::to_string(mouseCoords) << endl;
	}
}
