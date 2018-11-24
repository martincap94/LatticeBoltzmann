
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
#include <fstream>
#include <string>
#include <algorithm>

#include "LBM.h"
#include "LBM2D_1D_indices.h"
#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
#include "Grid2D.h"
#include "Grid3D.h"
#include "GeneralGrid.h"
#include "ShaderProgram.h"
#include "Camera.h"
#include "Camera2D.h"
#include "OrbitCamera.h"
#include "ParticleSystem.h"
#include "DirectionalLight.h"
#include "Grid.h"

int runApp();
void processInput(GLFWwindow* window);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void loadConfigFile();
void saveConfigParam(string param, string val);

enum LBMType {
	LBM2D,
	LBM3D,
	LBM2D_arr,
	LBM3D_arr
};

LBMType lbmType;

LBM *lbm;
Grid *grid;
Camera *camera;

int vsync = 0;
int numParticles = 1000; // default value
string sceneFilename;
bool useCUDA = true;

float deltaTime = 0.0f;
float lastFrameTime;

glm::mat4 view;
glm::mat4 projection;

int windowWidth = 1000;
int windowHeight = 1000;

int screenWidth;
int screenHeight;

int latticeWidth = 100;
int latticeHeight = 100;
int latticeDepth = 100;

float tau = 0.52f;




int main(int argc, char **argv) {
	srand(time(NULL));

	runApp();


}

int runApp() {

	loadConfigFile();
	//return 1; /////////////////////////////////////////////////////////////////

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	glfwWindowHint(GLFW_SAMPLES, 12); // enable MSAA with 4 samples

	GLFWwindow *window = glfwCreateWindow(windowWidth, windowHeight, "Lattice Boltzmann", nullptr, nullptr);

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


	glViewport(0, 0, screenWidth, screenHeight);

	float nearPlane = 0.1f;
	float farPlane = 1000.0f;

	float aspectRatio = screenWidth / screenHeight;



	ParticleSystem particles(numParticles);

	HeightMap heightMap(HEIGHTMAP_FILENAME, nullptr); // temporary fix, no need to load for 2D
	//HeightMap heightMap(sceneFilename, nullptr); // temporary fix, no need to load for 2D

	float projWidth;

	glm::vec3 dim(latticeWidth, latticeHeight, latticeDepth);

	switch (lbmType) {
		case LBM2D:
			printf("LBM2D SETUP...\n");
			lbm = new LBM2D_1D_indices(dim, tau, &particles);
			projWidth = (GRID_WIDTH > GRID_HEIGHT) ? GRID_WIDTH : GRID_HEIGHT;
			projection = glm::ortho(-1.0f, projWidth, -1.0f, projWidth, nearPlane, farPlane);
			grid = new Grid3D(6, 6, 6);
			camera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
			break;
		case LBM3D:
		default:
			printf("LBM3D SETUP...\n");
			lbm = new LBM3D_1D_indices(dim, tau, &particles, &heightMap);
			projection = glm::ortho(-50.0f, 50.0f, -50.0f, 50.0f, nearPlane, farPlane);
			grid = new Grid2D();
			camera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 10.0f, glm::vec3(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, GRID_DEPTH / 2.0f));
			break;
	}



	ShaderProgram singleColorShader("singleColor.vert", "singleColor.frag");
	ShaderProgram singleColorShaderAlpha("singleColor.vert", "singleColor_alpha.frag");
	ShaderProgram unlitColorShader("unlitColor.vert", "unlitColor.frag");
	ShaderProgram dirLightOnlyShader("dirLightOnly.vert", "dirLightOnly.frag");

	heightMap.shader = &dirLightOnlyShader;

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
	dirLightOnlyShader.setVec3("vViewPos", camera->position);



	glUseProgram(singleColorShader.id);
	singleColorShader.setMat4fv("projection", projection);
	glUseProgram(singleColorShaderAlpha.id);
	singleColorShaderAlpha.setMat4fv("projection", projection);
	glUseProgram(unlitColorShader.id);
	unlitColorShader.setMat4fv("uProjection", projection);

	glUseProgram(dirLightOnlyShader.id);
	dirLightOnlyShader.setMat4fv("uProjection", projection);

	GeneralGrid gGrid(100, 5);

	int frameCounter = 0;


	glEnable(GL_MULTISAMPLE); // enable multisampling (on by default)
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	glEnable(GL_DEPTH_TEST);

	glfwSwapInterval(vsync); // V-Sync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	float prevTime = glfwGetTime();

	while (!glfwWindowShouldClose(window)) {

		//cout << "frame " << frameCounter++ << endl;

		float currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		frameCounter++;
		if (currentFrameTime - prevTime >= 1.0f) {
			printf("Avg delta time = %0.4f [ms]\n", (1000.0f / frameCounter));
			prevTime += (currentFrameTime - prevTime);
			frameCounter = 0;
		}
		//cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;

		//glClearColor(0.1f, 0.4f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		


		glfwPollEvents();
		processInput(window);


		if (useCUDA) {
			lbm->doStepCUDA();
		} else {
			lbm->doStep();
		}


		view = camera->getViewMatrix();

		glUseProgram(singleColorShader.id);
		singleColorShader.setMat4fv("view", view);
		//singleColorShader.setMat4fv("projection", projection);
		
		glUseProgram(unlitColorShader.id);
		unlitColorShader.setMat4fv("uView", view);

		glUseProgram(dirLightOnlyShader.id);
		dirLightOnlyShader.setMat4fv("uView", view);
		dirLightOnlyShader.setVec3("vViewPos", camera->position);

		//unlitColorShader.setMat4fv("uProjection", projection);

		glUseProgram(singleColorShaderAlpha.id);
		singleColorShaderAlpha.setMat4fv("view", view);


		grid->draw(singleColorShader);

		lbm->draw(singleColorShader);



		particles.draw(singleColorShaderAlpha, useCUDA);

		//grid.draw(singleColorShaderAlpha);

		gGrid.draw(unlitColorShader);


		//camera.printInfo();



		glfwSwapBuffers(window);

	}

	delete lbm;
	delete grid;
	delete camera;

	glfwTerminate();
	return 0;
}




void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::UP, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::DOWN, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::LEFT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::ROTATE_RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::ROTATE_LEFT, deltaTime);
	}
#ifdef RUN_LBM3D
	if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_FRONT);
	}
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_SIDE);
	}	
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_TOP);
	}

#endif
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera->processMouseScroll(yoffset);
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

void loadConfigFile() {

	ifstream infile(CONFIG_FILE);

	string line;

	while (infile.good()) {

		getline(infile, line);

		// ignore comments
		if (line.find("//") == 0 || line.length() == 0) {
			continue;
		}
		// get rid of comments at the end of the line
		int idx = line.find("//");
		line = line.substr(0, idx);

		idx = line.find(":");
		// delete whitespace
		line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

		
		string param = line.substr(0, idx);
		string val = line.substr(idx + 1, line.length() - 1);

		//cout << "param = " << param << ", val = " << val << endl;
		cout << param << ": " << val << endl;

		saveConfigParam(param, val);


	}





}

void saveConfigParam(string param, string val) {

	if (param == "LBM_type") {
		if (val == "2D") {
			lbmType = LBM2D;
		} else if (val == "3D") {
			lbmType = LBM3D;
		} else if (val == "2D_arr") {
			lbmType = LBM2D_arr;
		} else if (val == "3D_arr") {
			lbmType = LBM3D_arr;
		}
	} else if (param == "VSync") {
		vsync = stoi(val);
	} else if (param == "num_particles") {
		numParticles = stoi(val);
	} else if (param == "scene_filename") {
		sceneFilename = val;
	} else if (param == "window_width") {
		windowWidth = stoi(val);
	} else if (param == "window_height") {
		windowHeight = stoi(val);
	} else if (param == "lattice_width") {
		latticeWidth = stoi(val);
	} else if (param == "lattice_height") {
		latticeHeight = stoi(val);
	} else if (param == "lattice_depth") {
		latticeDepth = stoi(val);
	} else if (param == "use_CUDA") {
		useCUDA = (val == "true") ? true : false;
	} else if (param == "tau") {
		tau = stof(val);
	}


}
