
#include <iostream>


#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "Config.h"

#define GLM_FORCE_CUDA // force GLM to be compatible with CUDA kernels
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
#include "Utils.h"

#include <omp.h>	// OpenMP for CPU parallelization

//#include <vld.h>	// Visual Leak Detector for memory leaks analysis

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

///// NUKLEAR /////////////////////////////////////////////////////////////////
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT
#include <nuklear.h>
#include "nuklear_glfw_gl3.h"

#define INCLUDE_STYLE
#ifdef INCLUDE_STYLE
#include "nuklear/style.c"
#endif

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024


///////////////////////////////////////////////////////////////////////////////////////////////////
///// FORWARD DECLARATIONS OF FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Run the application.
int runApp();

/// Process keyboard inputs of the window.
void processInput(GLFWwindow* window);

/// Mouse scroll callback for the window.
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

/// Mouse button callback for the window.
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

/// Load configuration file and parse all correct parameters.
void loadConfigFile();

/// Save configuration parameters to correct variables.
void saveConfigParam(string param, string val);

/// Constructs the user interface for the given context. Must be called in each frame!
void constructUserInterface(nk_context *ctx, nk_colorf &particlesColor);


///////////////////////////////////////////////////////////////////////////////////////////////////
///// ENUMS
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Unused enum that may be used to simplify shader usage.
enum eShaderProgram {
	DIR_LIGHT_SHADER,
	POINT_SPRITE_SHADER,
	SINGLE_COLOR_SHADER,
	SINGLE_COLOR_ALPHA_SHADER
};

/// Enum listing all possible LBM types. LBM2D_reindex and LBM3D_reindexed were deprecated, hence they are absent.
enum eLBMType {
	LBM2D,
	LBM3D
};



///////////////////////////////////////////////////////////////////////////////////////////////////
///// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////////////////////////

eLBMType lbmType;		///< The LBM type that is to be displayed

LBM *lbm;				///< Pointer to the current LBM
Grid *grid;				///< Pointer to the current grid
Camera *camera;			///< Pointer to the current camera
ParticleSystem *particleSystem;		///< Pointer to the particle system that is to be used throughout the whole application

///////////////////////////////////////////////////////////////////////////////////////////////////
///// DEFAULT VALUES THAT ARE TO BE REWRITTEN FROM THE CONFIG FILE
/////	-> this should be moved to more convenient/readable location
/////		(helper namespace or class that is used only for configuration...)
///////////////////////////////////////////////////////////////////////////////////////////////////
int vsync = 0;				///< VSync value
int numParticles = 1000;	///< Number of particles
string sceneFilename;		///< Filename of the scene
bool useCUDA = true;		///< Whether to use CUDA or run the CPU version of the application
int useCUDACheckbox = 1;	///< Helper int value for the UI checkbox

float deltaTime = 0.0f;		///< Delta time of the current frame
float lastFrameTime;		///< Duration of the last frame

glm::mat4 view;				///< View matrix
glm::mat4 projection;		///< Projection matrix

int windowWidth = 1000;		///< Window width
int windowHeight = 1000;	///< Window height

int screenWidth;			///< Screen width
int screenHeight;			///< Screen height

int latticeWidth = 100;		///< Default lattice width
int latticeHeight = 100;	///< Default lattice height
int latticeDepth = 100;		///< Defailt lattice depth

float tau = 0.52f;			///< Default tau value

bool drawStreamlines = false;	///< Whether to draw streamlines - DRAWING STREAMLINES CURRENTLY NOT VIABLE
int paused = 0;				///< Whether the simulation is paused
int usePointSprites = 0;	///< Whether to use point sprites for point visualization
bool appRunning = true;		///< Helper boolean to stop the application with the exit button in the user interface
float cameraSpeed = DEFAULT_CAMERA_SPEED;	///< Movement speed of the main camera

int blockDim_2D = 256;		///< Block dimension for 2D LBM
int blockDim_3D_x = 32;		///< Block x dimension for 3D LBM
int blockDim_3D_y = 2;		///< Block y dimension for 3D LBM


/// Main - runs the application and sets seed for the random number generator.
int main(int argc, char **argv) {
	srand(time(NULL));

	runApp();

	return 0;
}



/// Runs the application including the game loop.
/**
	Creates the window, user interface and all the main parts of the simulation including the simulator itself (either
	LBM 2D or 3D), grids, particle system, and collider object (2D) or height map (3D).
	Furthermore, creates all the shaders and runs the main game loop of the application in which the simulation is updated
	and the UI is drawn (and constructed since nuklear panel needs to be constructed in each frame).
*/
int runApp() {

	loadConfigFile();

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


	glViewport(0, 0, screenWidth, screenHeight);

	float nearPlane = 0.1f;
	float farPlane = 1000.0f;

	float aspectRatio = screenWidth / screenHeight;

	struct nk_context *ctx = nk_glfw3_init(window, NK_GLFW3_INSTALL_CALLBACKS);

	{
		struct nk_font_atlas *atlas;
		nk_glfw3_font_stash_begin(&atlas);
		struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/Roboto-Regular.ttf", 14, 0);
		nk_glfw3_font_stash_end();
		nk_style_load_all_cursors(ctx, atlas->cursors);
		nk_style_set_font(ctx, &roboto->handle);
	}

#ifdef INCLUDE_STYLE
	set_style(ctx, THEME_MARTIN); // this theme uses transparent background so we can see the simulation through the UI
#endif

	struct nk_colorf particlesColor;

	particleSystem = new ParticleSystem(numParticles, drawStreamlines);

	particlesColor.r = particleSystem->particlesColor.r;
	particlesColor.g = particleSystem->particlesColor.g;
	particlesColor.b = particleSystem->particlesColor.b;


	float projWidth;

	glm::vec3 latticeDim(latticeWidth, latticeHeight, latticeDepth);

	// Create and configure the simulator, select from 2D and 3D options and set parameters accordingly
	switch (lbmType) {
		case LBM2D:
			printf("LBM2D SETUP...\n");
			lbm = new LBM2D_1D_indices(latticeDim, sceneFilename, tau, particleSystem, blockDim_2D);

			latticeWidth = lbm->latticeWidth;
			latticeHeight = lbm->latticeHeight;
			latticeDepth = 1;

			projWidth = (latticeWidth > latticeHeight) ? latticeWidth : latticeHeight;
			projection = glm::ortho(-1.0f, projWidth, -1.0f, projWidth, nearPlane, farPlane);
			//projection = glm::ortho(-1.0f, (float)latticeWidth, -1.0f, (float)latticeHeight, nearPlane, farPlane);
			grid = new Grid2D(latticeWidth, latticeHeight, latticeWidth / 100, latticeWidth / 100);
			camera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
			break;
		case LBM3D:
		default:
			printf("LBM3D SETUP...\n");

			dim3 blockDim(blockDim_3D_x, blockDim_3D_y, 1);

			lbm = new LBM3D_1D_indices(latticeDim, sceneFilename, tau, particleSystem, blockDim);

			latticeWidth = lbm->latticeWidth;
			latticeHeight = lbm->latticeHeight;
			latticeDepth = lbm->latticeDepth;

			float projectionRange = (latticeWidth > latticeHeight) ? latticeWidth : latticeHeight;
			projectionRange = (projectionRange > latticeDepth) ? projectionRange : latticeDepth;
			projectionRange /= 2.0f;

			projection = glm::ortho(-projectionRange, projectionRange, -projectionRange, projectionRange, nearPlane, farPlane);
			grid = new Grid3D(latticeWidth, latticeHeight, latticeDepth, 6, 6, 6);
			camera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 80.0f, glm::vec3(latticeWidth / 2.0f, latticeHeight / 2.0f, latticeDepth / 2.0f));
			break;
	}
	camera->setLatticeDimensions(latticeWidth, latticeHeight, latticeDepth);
	camera->movementSpeed = cameraSpeed;
	particleSystem->lbm = lbm;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	///// SHADERS - should be simplified with helper classes, unfortunately due to time constraints
	/////			it has remained in this form
	//////////////////////////////////////////////////////////////////////////////////////////////////
	ShaderProgram singleColorShader("singleColor.vert", "singleColor.frag");
	ShaderProgram singleColorShaderAlpha("singleColor.vert", "singleColor_alpha.frag");
	ShaderProgram unlitColorShader("unlitColor.vert", "unlitColor.frag");
	ShaderProgram dirLightOnlyShader("dirLightOnly.vert", "dirLightOnly.frag");
	ShaderProgram pointSpriteTestShader("pointSpriteTest.vert", "pointSpriteTest.frag");
	ShaderProgram coloredParticleShader("coloredParticle.vert", "coloredParticle.frag");

	if (lbmType == LBM3D) {
		((LBM3D_1D_indices*)lbm)->heightMap->shader = &dirLightOnlyShader;
	}

	DirectionalLight dirLight;
	dirLight.direction = glm::vec3(41.0f, 45.0f, 1.0f);
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
	singleColorShader.setMat4fv("uProjection", projection);
	glUseProgram(singleColorShaderAlpha.id);
	singleColorShaderAlpha.setMat4fv("uProjection", projection);
	glUseProgram(unlitColorShader.id);
	unlitColorShader.setMat4fv("uProjection", projection);

	glUseProgram(dirLightOnlyShader.id);
	dirLightOnlyShader.setMat4fv("uProjection", projection);
	glUseProgram(pointSpriteTestShader.id);
	pointSpriteTestShader.setMat4fv("uProjection", projection);

	glUseProgram(coloredParticleShader.id);
	coloredParticleShader.setMat4fv("uProjection", projection);

	GeneralGrid gGrid(100, 5);

	int frameCounter = 0;
	glfwSwapInterval(vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)
	
	float prevTime = glfwGetTime();
	int totalFrameCounter = 0;

	// Set these callbacks after nuklear initialization, otherwise they won't work!
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	while (!glfwWindowShouldClose(window) && appRunning) {

		// enable flags each frame because nuklear disables them when it is rendered
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_TEXTURE_1D);
		glEnable(GL_TEXTURE_3D);

		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		float currentFrameTime = glfwGetTime();
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;
		frameCounter++;
		totalFrameCounter++;
		if (currentFrameTime - prevTime >= 1.0f) {
			printf("Avg delta time = %0.4f [ms]\n", (1000.0f / frameCounter));
			prevTime += (currentFrameTime - prevTime);
			frameCounter = 0;
		}
		//cout << " Delta time = " << (deltaTime * 1000.0f) << " [ms]" << endl;
		//cout << " Framerate = " << (1.0f / deltaTime) << endl;


		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwPollEvents();
		processInput(window);
		constructUserInterface(ctx, particlesColor);

		// MAIN SIMULATION STEP
		if (!paused) {
			if (useCUDA) {
				lbm->doStepCUDA();
			} else {
				lbm->doStep();
			}
		}


		// UPDATE SHADER VIEW MATRICES
		view = camera->getViewMatrix();

		glUseProgram(singleColorShader.id);
		singleColorShader.setMat4fv("uView", view);
		//singleColorShader.setMat4fv("projection", projection);

		glUseProgram(unlitColorShader.id);
		unlitColorShader.setMat4fv("uView", view);

		glUseProgram(dirLightOnlyShader.id);
		dirLightOnlyShader.setMat4fv("uView", view);
		dirLightOnlyShader.setVec3("vViewPos", camera->position);

		//unlitColorShader.setMat4fv("uProjection", projection);

		glUseProgram(singleColorShaderAlpha.id);
		singleColorShaderAlpha.setMat4fv("uView", view);

		glUseProgram(pointSpriteTestShader.id);
		pointSpriteTestShader.setMat4fv("uView", view);

		glUseProgram(coloredParticleShader.id);
		coloredParticleShader.setMat4fv("uView", view);



		// DRAW SCENE
		grid->draw(singleColorShader);
		lbm->draw(singleColorShader);

		if (usePointSprites) {
			particleSystem->draw(pointSpriteTestShader, useCUDA);
		} else if (lbm->visualizeVelocity) {
			particleSystem->draw(coloredParticleShader, useCUDA);
		} else {
			particleSystem->draw(singleColorShader, useCUDA);
		}
		gGrid.draw(unlitColorShader);

		// Render the user interface
		nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);

		lbm->recalculateVariables(); // recalculate variables based on values set in the user interface

		glfwSwapBuffers(window);

	}


	delete particleSystem;
	delete lbm;
	delete grid;
	delete camera;



	size_t cudaMemFree = 0;
	size_t cudaMemTotal = 0;

	cudaMemGetInfo(&cudaMemFree, &cudaMemTotal);

	/*
	cout << " FREE CUDA MEMORY  = " << cudaMemFree << endl;
	cout << " TOTAL CUDA MEMORY = " << cudaMemTotal << endl;
	*/


	nk_glfw3_shutdown();
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
		camera->processKeyboardMovement(Camera::ROTATE_LEFT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::ROTATE_RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_FRONT);
	}
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_SIDE);
	}
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		camera->setView(Camera::VIEW_TOP);
	}
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
		lbm->resetSimulation();
	}
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

		// delete whitespace
		trim(line);
		//line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

		idx = line.find(":");

		string param = line.substr(0, idx);
		string val = line.substr(idx + 1, line.length() - 1);
		trim(param);
		trim(val);

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
		useCUDACheckbox = (int)useCUDA;
	} else if (param == "tau") {
		tau = stof(val);
	} else if (param == "draw_streamlines") {
		drawStreamlines = (val == "true") ? true : false;
	} else if (param == "autoplay") {
		paused = (val == "true") ? 0 : 1;
	} else if (param == "camera_speed") {
		cameraSpeed = stof(val);
	} else if (param == "block_dim_2D") {
		blockDim_2D = stoi(val);
	} else if (param == "block_dim_3D_x") {
		blockDim_3D_x = stoi(val);
	} else if (param == "block_dim_3D_y") {
		blockDim_3D_y = stoi(val);
	}
}

void constructUserInterface(nk_context *ctx, nk_colorf &particlesColor) {
	nk_glfw3_new_frame();

	/* GUI */
	if (nk_begin(ctx, "Control Panel", nk_rect(50, 50, 275, 350),
				 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
				 NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {
		enum { EASY, HARD };
		static int op = EASY;
		static int property = 20;
		nk_layout_row_static(ctx, 30, 80, 3);
		if (nk_button_label(ctx, "Reset")) {
			//fprintf(stdout, "button pressed\n");
			lbm->resetSimulation();
		}
		const char *buttonDescription = paused ? "Play" : "Pause";
		if (nk_button_label(ctx, buttonDescription)) {
			paused = !paused;
		}
		if (nk_button_label(ctx, "EXIT")) {
			appRunning = false;
		}


#ifdef LBM_EXPERIMENTAL
		nk_layout_row_dynamic(ctx, 30, 1);
		nk_label_colored_wrap(ctx, "Enabling or disabling CUDA at runtime is highly unstable at the moment, use at your own discretion", nk_rgba_f(1.0f, 0.5f, 0.5f, 1.0f));

		bool useCUDAPrev = useCUDA;
		nk_checkbox_label(ctx, "Use CUDA", &useCUDACheckbox);
		useCUDA = useCUDACheckbox;
		if (useCUDAPrev != useCUDA && useCUDA == false) {
			lbm->switchToCPU();
		}
#endif

		nk_layout_row_dynamic(ctx, 25, 1);

		nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 10.0f, 0.005f, 0.005f);

		int mirrorSidesPrev = lbm->mirrorSides;
		nk_layout_row_dynamic(ctx, 15, 1);
		nk_checkbox_label(ctx, "Mirror sides", &lbm->mirrorSides);
		if (mirrorSidesPrev != lbm->mirrorSides) {
			cout << "Mirror sides value changed!" << endl;
			lbm->updateControlProperty(LBM::MIRROR_SIDES_PROP);
		}

		if (lbmType == LBM3D) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_checkbox_label(ctx, "Use subgrid model", &lbm->useSubgridModel);
		}



		nk_layout_row_dynamic(ctx, 15, 1);
		//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
		int prevVsync = vsync;
		nk_checkbox_label(ctx, "VSync", &vsync);
		if (prevVsync != vsync) {
			glfwSwapInterval(vsync);
		}

		nk_label(ctx, "Inlet velocity:", NK_TEXT_LEFT);
				
		nk_layout_row_dynamic(ctx, 15, (lbmType == LBM2D) ? 2 : 3);
		nk_property_float(ctx, "x:", 0.0f, &lbm->inletVelocity.x, 1.0f, 0.01f, 0.005f);
		nk_property_float(ctx, "y:", -1.0f, &lbm->inletVelocity.y, 1.0f, 0.01f, 0.005f);
		if (lbmType == LBM3D) {
			nk_property_float(ctx, "z:", -1.0f, &lbm->inletVelocity.z, 1.0f, 0.01f, 0.005f);
		}


		nk_layout_row_dynamic(ctx, 15, 1);
		//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
		nk_checkbox_label(ctx, "Use point sprites", &usePointSprites);

		if (lbmType == LBM2D && useCUDA && !usePointSprites) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_checkbox_label(ctx, "Visualize velocity", &lbm->visualizeVelocity);
		}

		nk_layout_row_dynamic(ctx, 10, 1);
		nk_labelf(ctx, NK_TEXT_LEFT, "Point size");
		nk_slider_int(ctx, 1, &particleSystem->pointSize, 100, 1);

		if (!usePointSprites && !lbm->visualizeVelocity) {
			nk_layout_row_dynamic(ctx, 20, 1);
			nk_label(ctx, "Particles Color:", NK_TEXT_LEFT);
			nk_layout_row_dynamic(ctx, 25, 1);
			if (nk_combo_begin_color(ctx, nk_rgb_cf(particlesColor), nk_vec2(nk_widget_width(ctx), 400))) {
				nk_layout_row_dynamic(ctx, 120, 1);
				particlesColor = nk_color_picker(ctx, particlesColor, NK_RGBA);
				nk_layout_row_dynamic(ctx, 25, 1);
				particlesColor.r = nk_propertyf(ctx, "#R:", 0, particlesColor.r, 1.0f, 0.01f, 0.005f);
				particlesColor.g = nk_propertyf(ctx, "#G:", 0, particlesColor.g, 1.0f, 0.01f, 0.005f);
				particlesColor.b = nk_propertyf(ctx, "#B:", 0, particlesColor.b, 1.0f, 0.01f, 0.005f);
				particlesColor.a = nk_propertyf(ctx, "#A:", 0, particlesColor.a, 1.0f, 0.01f, 0.005f);
				particleSystem->particlesColor = glm::vec3(particlesColor.r, particlesColor.g, particlesColor.b);
				nk_combo_end(ctx);
			}
		}

		nk_layout_row_dynamic(ctx, 15, 1);
		nk_label(ctx, "Camera movement speed", NK_TEXT_LEFT);
		nk_slider_float(ctx, 1.0f, &camera->movementSpeed, 400.0f, 1.0f);



	}
	nk_end(ctx);


}
