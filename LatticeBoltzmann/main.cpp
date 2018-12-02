
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

//#include <vld.h>

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

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024


int runApp();
void processInput(GLFWwindow* window);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void loadConfigFile();
void saveConfigParam(string param, string val);
void constructUserInterface(nk_context *ctx, nk_colorf &bg);

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
ParticleSystem *particleSystem;

int vsync = 0;
int numParticles = 1000; // default value
string sceneFilename;
bool useCUDA = true;
int useCUDACheckbox = 1;

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

bool drawStreamlines = false;

int paused = 0;



int main(int argc, char **argv) {
	srand(time(NULL));

	runApp();

	return 0;
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

	//if (lbmType == LBM2D) {
	//	float ratio = (float)latticeWidth / (float)latticeHeight;
	//	cout << "RATIO = " << ratio << endl;
	//	windowHeight /= ratio;
	//}

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

	struct nk_context *ctx = nk_glfw3_init(window, NK_GLFW3_INSTALL_CALLBACKS);

	{
		struct nk_font_atlas *atlas;
		nk_glfw3_font_stash_begin(&atlas);
		/*struct nk_font *droid = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/DroidSans.ttf", 14, 0);*/
		struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/Roboto-Regular.ttf", 14, 0);
		/*struct nk_font *future = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/kenvector_future_thin.ttf", 13, 0);*/
		/*struct nk_font *clean = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/ProggyClean.ttf", 12, 0);*/
		/*struct nk_font *tiny = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/ProggyTiny.ttf", 10, 0);*/
		/*struct nk_font *cousine = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/Cousine-Regular.ttf", 13, 0);*/
		nk_glfw3_font_stash_end();
		nk_style_load_all_cursors(ctx, atlas->cursors);
		//nk_style_set_font(ctx, &droid->handle);
		nk_style_set_font(ctx, &roboto->handle);
	}

#ifdef INCLUDE_STYLE
	/*set_style(ctx, THEME_WHITE);*/
	/*set_style(ctx, THEME_RED);*/
	/*set_style(ctx, THEME_BLUE);*/
	/*set_style(ctx, THEME_DARK);*/
#endif
	struct nk_colorf bg;
	//bg.r = 0.1f;
	//bg.g = 0.18f;
	//bg.b = 0.24f;
	//bg.a = 1.0f;

	particleSystem = new ParticleSystem(numParticles, drawStreamlines);

	bg.r = particleSystem->particlesColor.r;
	bg.g = particleSystem->particlesColor.g;
	bg.b = particleSystem->particlesColor.b;
	//bg.a = particleSystem->particlesColor.a;


	//HeightMap heightMap(HEIGHTMAP_FILENAME, nullptr); // temporary fix, no need to load for 2D
	//HeightMap heightMap(sceneFilename, nullptr); // temporary fix, no need to load for 2D

	float projWidth;

	glm::vec3 dim(latticeWidth, latticeHeight, latticeDepth);

	switch (lbmType) {
		case LBM2D:
			printf("LBM2D SETUP...\n");
			lbm = new LBM2D_1D_indices(dim, sceneFilename, tau, particleSystem);

			latticeWidth = lbm->latticeWidth;
			latticeHeight = lbm->latticeHeight;
			latticeDepth = 1;

			projWidth = (latticeWidth > latticeHeight) ? latticeWidth : latticeHeight;
			projection = glm::ortho(-1.0f, projWidth, -1.0f, projWidth, nearPlane, farPlane);
			//projection = glm::ortho(-1.0f, (float)latticeWidth, -1.0f, (float)latticeHeight, nearPlane, farPlane);
			grid = new Grid2D();
			camera = new Camera2D(glm::vec3(0.0f, 0.0f, 100.0f), WORLD_UP, -90.0f, 0.0f);
			break;
		case LBM3D:
		default:
			printf("LBM3D SETUP...\n");
			lbm = new LBM3D_1D_indices(dim, sceneFilename, tau, particleSystem);

			latticeWidth = lbm->latticeWidth;
			latticeHeight = lbm->latticeHeight;
			latticeDepth = lbm->latticeDepth;

			float projectionRange = (latticeWidth > latticeHeight) ? latticeWidth : latticeHeight;
			projectionRange = (projectionRange > latticeDepth) ? projectionRange : latticeDepth;
			projectionRange /= 2.0f;

			projection = glm::ortho(-projectionRange, projectionRange, -projectionRange, projectionRange, nearPlane, farPlane);
			grid = new Grid3D(6, 6, 6);
			camera = new OrbitCamera(glm::vec3(0.0f, 0.0f, 0.0f), WORLD_UP, 45.0f, 10.0f, glm::vec3(latticeWidth / 2.0f, latticeHeight / 2.0f, latticeDepth / 2.0f));
			break;
	}
	camera->setLatticeDimensions(latticeWidth, latticeHeight, latticeDepth);



	ShaderProgram singleColorShader("singleColor.vert", "singleColor.frag");
	ShaderProgram singleColorShaderAlpha("singleColor.vert", "singleColor_alpha.frag");
	ShaderProgram unlitColorShader("unlitColor.vert", "unlitColor.frag");
	ShaderProgram dirLightOnlyShader("dirLightOnly.vert", "dirLightOnly.frag");

	if (lbmType == LBM3D) {
		((LBM3D_1D_indices*)lbm)->testHM->shader = &dirLightOnlyShader;
	}

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

	glfwSwapInterval(vsync); // VSync Settings (0 is off, 1 is 60FPS, 2 is 30FPS and so on)

	float prevTime = glfwGetTime();

	int totalFrameCounter = 0;

	while (!glfwWindowShouldClose(window)) {

		// enable flags because of nuklear
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_TEXTURE_1D);
		glEnable(GL_TEXTURE_3D);

		//cout << "frame " << frameCounter++ << endl;

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

		//glClearColor(0.1f, 0.4f, 0.4f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



		glfwPollEvents();

		processInput(window);

		constructUserInterface(ctx, bg);

		if (!paused) {
			if (useCUDA) {
				lbm->doStepCUDA();
			} else {
				lbm->doStep();
			}
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



		particleSystem->draw(singleColorShader, useCUDA);

		//grid.draw(singleColorShaderAlpha);

		gGrid.draw(unlitColorShader);


		//camera.printInfo();


		nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
		lbm->recalculateVariables();

		glfwSwapBuffers(window);

	}

	delete lbm;
	delete grid;
	delete camera;
	delete particleSystem;

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
		camera->processKeyboardMovement(Camera::ROTATE_RIGHT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		camera->processKeyboardMovement(Camera::ROTATE_LEFT, deltaTime);
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
		useCUDACheckbox = (int)useCUDA;
	} else if (param == "tau") {
		tau = stof(val);
	} else if (param == "draw_streamlines") {
		drawStreamlines = (val == "true") ? true : false;
	} else if (param == "autoplay") {
		paused = (val == "true") ? 0 : 1;
	}


}

void constructUserInterface(nk_context *ctx, nk_colorf &bg) {
	nk_glfw3_new_frame();

	/* GUI */
	if (nk_begin(ctx, "Control Panel", nk_rect(50, 50, 230, 250),
				 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
				 NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {
		enum { EASY, HARD };
		static int op = EASY;
		static int property = 20;
		nk_layout_row_static(ctx, 30, 80, 1);
		if (nk_button_label(ctx, "Reset")) {
			//fprintf(stdout, "button pressed\n");
			lbm->resetSimulation();
		}
		const char *buttonDescription = paused ? "Play" : "Pause";
		if (nk_button_label(ctx, buttonDescription)) {
			paused = !paused;
		}



		nk_layout_row_dynamic(ctx, 15, 1);
		//if (nk_option_label(ctx, "easy", op == EASY)) op = EASY;
		//if (nk_option_label(ctx, "hard", op == HARD)) op = HARD;
		nk_label_colored_wrap(ctx, "Enabling or disabling CUDA at runtime is highly unstable at the moment, use at your own discretion", nk_rgba_f(1.0f, 0.5f, 0.5f, 1.0f));
		nk_checkbox_label(ctx, "Use CUDA", &useCUDACheckbox);
		//cout << (bool)useCUDACheckbox << endl;
		useCUDA = useCUDACheckbox;

		nk_layout_row_dynamic(ctx, 25, 1);
		//nk_property_int(ctx, "Compression:", 0, &property, 100, 10, 1);

		//nk_label(ctx, "(CHANGING THIS VALUE AT RUNTIME IS EXTREMELY UNSTABLE)", NK_TEXT_LEFT);
		nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 2.0f, 0.005f, 0.005f);
		//cout << "tau set to " << lbm->tau << endl;

		nk_layout_row_dynamic(ctx, 20, 1);
		nk_label(ctx, "Particles Color:", NK_TEXT_LEFT);
		nk_layout_row_dynamic(ctx, 25, 1);
		if (nk_combo_begin_color(ctx, nk_rgb_cf(bg), nk_vec2(nk_widget_width(ctx), 400))) {
			nk_layout_row_dynamic(ctx, 120, 1);
			bg = nk_color_picker(ctx, bg, NK_RGBA);
			nk_layout_row_dynamic(ctx, 25, 1);
			bg.r = nk_propertyf(ctx, "#R:", 0, bg.r, 1.0f, 0.01f, 0.005f);
			bg.g = nk_propertyf(ctx, "#G:", 0, bg.g, 1.0f, 0.01f, 0.005f);
			bg.b = nk_propertyf(ctx, "#B:", 0, bg.b, 1.0f, 0.01f, 0.005f);
			bg.a = nk_propertyf(ctx, "#A:", 0, bg.a, 1.0f, 0.01f, 0.005f);
			particleSystem->particlesColor = glm::vec3(bg.r, bg.g, bg.b);
			nk_combo_end(ctx);
		}
	}
	nk_end(ctx);


}
