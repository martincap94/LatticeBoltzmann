#include "Utils.h"

#include "DataStructures.h"

//float calculateMacroscopicDensity(int x, int y, int z) {
//	float macroDensity = 0.0f;
//	int idx = getIdx(x, y, z);
//	for (int i = 0; i < 19; i++) {
//		macroDensity += backLattice[idx].adj[i];
//	}
//	return macroDensity;
//}
//
//glm::vec3 calculateMacroscopicVelocity(int x, int y, int z, float macroDensity) {
//
//	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
//
//	int idx = getIdx(x, y, z);
//	//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
//	macroVelocity += vLeft * backLattice[idx].adj[DIR_LEFT_FACE];
//	macroVelocity += vFront * backLattice[idx].adj[DIR_FRONT_FACE];
//	macroVelocity += vBottom * backLattice[idx].adj[DIR_BOTTOM_FACE];
//	macroVelocity += vFrontLeft * backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
//	macroVelocity += vBackLeft * backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
//	macroVelocity += vBottomLeft * backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];
//	macroVelocity += vTopLeft * backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
//	macroVelocity += vBottomFront * backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
//	macroVelocity += vTopFront * backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
//	macroVelocity += vRight * backLattice[idx].adj[DIR_RIGHT_FACE];
//	macroVelocity += vBack * backLattice[idx].adj[DIR_BACK_FACE];
//	macroVelocity += vTop * backLattice[idx].adj[DIR_TOP_FACE];
//	macroVelocity += vBackRight * backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
//	macroVelocity += vFrontRight * backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
//	macroVelocity += vTopRight * backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
//	macroVelocity += vBottomRight * backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
//	macroVelocity += vTopBack * backLattice[idx].adj[DIR_TOP_BACK_EDGE];
//	macroVelocity += vBottomBack * backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
//	macroVelocity /= macroDensity;
//
//	return macroVelocity;
//}
