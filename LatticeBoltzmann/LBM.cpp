#include "LBM.h"


LBM::LBM() {
}

LBM::LBM(glm::vec3 dimensions, float tau) : latticeWidth(dimensions.x), latticeHeight(dimensions.y), latticeDepth(dimensions.z), tau(tau) {
	itau = 1.0f / tau;
}


LBM::~LBM() {
}
