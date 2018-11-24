#include "LBM.h"


LBM::LBM() {
}

LBM::LBM(glm::vec3 dimensions, string sceneFilename, float tau) : latticeWidth(dimensions.x), latticeHeight(dimensions.y), latticeDepth(dimensions.z), sceneFilename(sceneFilename), tau(tau) {
	itau = 1.0f / tau;
	nu = (2.0f * tau - 1.0f) / 6.0f;
}


LBM::~LBM() {
}
