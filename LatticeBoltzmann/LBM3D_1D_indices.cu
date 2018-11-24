#include "LBM3D_1D_indices.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>



__constant__ int d_latticeWidth;
__constant__ int d_latticeHeight;
__constant__ int d_latticeDepth;
__constant__ int d_latticeSize;
__constant__ float d_tau;
__constant__ float d_itau;


__device__ int getIdxKer(int x, int y, int z) {
	return (x + d_latticeWidth * (y + d_latticeHeight * z));
}


__global__ void moveParticlesKernelInterop(float3 *particleVertices, glm::vec3 *velocities, int *numParticles) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	glm::vec3 adjVelocities[8];

	while (idx < *numParticles) {

		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;
		float z = particleVertices[idx].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[getIdxKer(leftX, topY, backZ)];
		adjVelocities[1] = velocities[getIdxKer(rightX, topY, backZ)];
		adjVelocities[2] = velocities[getIdxKer(leftX, bottomY, backZ)];
		adjVelocities[3] = velocities[getIdxKer(rightX, bottomY, backZ)];
		adjVelocities[4] = velocities[getIdxKer(leftX, topY, frontZ)];
		adjVelocities[5] = velocities[getIdxKer(rightX, topY, frontZ)];
		adjVelocities[6] = velocities[getIdxKer(leftX, bottomY, frontZ)];
		adjVelocities[7] = velocities[getIdxKer(rightX, bottomY, frontZ)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;
		float depthRatio = z - backZ;

		glm::vec3 topBackVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec3 bottomBackVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec3 backVelocity = bottomBackVelocity * verticalRatio + topBackVelocity * (1.0f - verticalRatio);

		glm::vec3 topFrontVelocity = adjVelocities[4] * horizontalRatio + adjVelocities[5] * (1.0f - horizontalRatio);
		glm::vec3 bottomFrontVelocity = adjVelocities[6] * horizontalRatio + adjVelocities[7] * (1.0f - horizontalRatio);

		glm::vec3 frontVelocity = bottomFrontVelocity * verticalRatio + topFrontVelocity * (1.0f - verticalRatio);

		glm::vec3 finalVelocity = backVelocity * depthRatio + frontVelocity * (1.0f - depthRatio);

		particleVertices[idx].x += finalVelocity.x;
		particleVertices[idx].y += finalVelocity.y;
		particleVertices[idx].z += finalVelocity.z;


		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
			particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 ||
			particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {

			if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1) {
				particleVertices[idx].x = 0.0f;
			} else {
				particleVertices[idx].y = (float)((int)(particleVertices[idx].y + d_latticeHeight - 1) % (d_latticeHeight - 1));
				particleVertices[idx].z = (float)((int)(particleVertices[idx].z + d_latticeDepth - 1) % (d_latticeDepth - 1));
			}
			/*particleVertices[idx].x = 0.0f;
			particleVertices[idx].y = 0.0f;
			particleVertices[idx].z = 0.0f;*/
		}
		idx += blockDim.x * blockDim.y * gridDim.x;

	}
}


__global__ void moveParticlesKernel(glm::vec3 *particleVertices, glm::vec3 *velocities, int *numParticles) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	glm::vec3 adjVelocities[8];

	while (idx < *numParticles) {

		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;
		float z = particleVertices[idx].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[getIdxKer(leftX, topY, backZ)];
		adjVelocities[1] = velocities[getIdxKer(rightX, topY, backZ)];
		adjVelocities[2] = velocities[getIdxKer(leftX, bottomY, backZ)];
		adjVelocities[3] = velocities[getIdxKer(rightX, bottomY, backZ)];
		adjVelocities[4] = velocities[getIdxKer(leftX, topY, frontZ)];
		adjVelocities[5] = velocities[getIdxKer(rightX, topY, frontZ)];
		adjVelocities[6] = velocities[getIdxKer(leftX, bottomY, frontZ)];
		adjVelocities[7] = velocities[getIdxKer(rightX, bottomY, frontZ)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;
		float depthRatio = z - backZ;

		glm::vec3 topBackVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec3 bottomBackVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec3 backVelocity = bottomBackVelocity * verticalRatio + topBackVelocity * (1.0f - verticalRatio);

		glm::vec3 topFrontVelocity = adjVelocities[4] * horizontalRatio + adjVelocities[5] * (1.0f - horizontalRatio);
		glm::vec3 bottomFrontVelocity = adjVelocities[6] * horizontalRatio + adjVelocities[7] * (1.0f - horizontalRatio);

		glm::vec3 frontVelocity = bottomFrontVelocity * verticalRatio + topFrontVelocity * (1.0f - verticalRatio);

		glm::vec3 finalVelocity = backVelocity * depthRatio + frontVelocity * (1.0f - depthRatio);

		particleVertices[idx] += finalVelocity;


		//if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
		//	particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 ||
		//	particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {

		//	int respawnX = idx % d_latticeHeight;
		//	int respawnY = (idx / d_latticeHeight) % d_latticeWidth;
		//	int respawnZ = idx / (d_latticeWidth * d_latticeHeight);

		//	particleVertices[idx] = glm::vec3(0, respawnY, respawnZ);
		//}
		idx += blockDim.x * blockDim.y * gridDim.x;

	}
}


__global__ void clearBackLatticeKernel(Node3D *backLattice) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {
		for (int i = 0; i < 19; i++) {
			backLattice[idx].adj[i] = 0.0f;
		}
	}
}

// ineffective - rewrite so not all threads have to compute the same stuff, just testing for now
__global__ void updateInletsKernel(Node3D *backLattice, glm::vec3 *velocities) {

	float macroDensity = 1.0f;
	glm::vec3 macroVelocity = glm::vec3(0.4f, 0.0f, 0.0f);


	float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
	float leftTermAxis = WEIGHT_AXIS * macroDensity;
	float leftTermNonaxial = WEIGHT_NON_AXIAL * macroDensity;


	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	// old and incorrect?
	//int x = idx % d_latticeHeight;
	//int y = (idx / d_latticeHeight) % d_latticeWidth;
	//int z = idx / (d_latticeWidth * d_latticeHeight);

	int x = idx % d_latticeWidth;
	//int y = (idx / d_latticeWidth) % d_latticeHeight;
	//int z = idx / (d_latticeHeight * d_latticeWidth);


	if (x == 0 && idx < d_latticeSize) {
		backLattice[idx].adj[DIR_MIDDLE_VERTEX] = middleEq;
		backLattice[idx].adj[DIR_RIGHT_FACE] = rightEq;
		backLattice[idx].adj[DIR_LEFT_FACE] = leftEq;
		backLattice[idx].adj[DIR_BACK_FACE] = backEq;
		backLattice[idx].adj[DIR_FRONT_FACE] = frontEq;
		backLattice[idx].adj[DIR_TOP_FACE] = topEq;
		backLattice[idx].adj[DIR_BOTTOM_FACE] = bottomEq;
		backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = backRightEq;
		backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = backLeftEq;
		backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = frontRightEq;
		backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = frontLeftEq;
		backLattice[idx].adj[DIR_TOP_BACK_EDGE] = topBackEq;
		backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = topFrontEq;
		backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = bottomBackEq;
		backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = bottomFrontEq;
		backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = topRightEq;
		backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = topLeftEq;
		backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = bottomRightEq;
		backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = bottomLeftEq;


		for (int i = 0; i < 19; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}
}


__global__ void collisionStepKernel(Node3D *backLattice, glm::vec3 *velocities) {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {
		float macroDensity = 0.0f;
		for (int i = 0; i < 19; i++) {
			macroDensity += backLattice[idx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
		macroVelocity += dirVectorsConst[DIR_LEFT_FACE] * backLattice[idx].adj[DIR_LEFT_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_FACE] * backLattice[idx].adj[DIR_FRONT_FACE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FACE] * backLattice[idx].adj[DIR_BOTTOM_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_LEFT_EDGE] * backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BACK_LEFT_EDGE] * backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_LEFT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_LEFT_EDGE] * backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FRONT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_FRONT_EDGE] * backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_RIGHT_FACE] * backLattice[idx].adj[DIR_RIGHT_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_FACE] * backLattice[idx].adj[DIR_BACK_FACE];
		macroVelocity += dirVectorsConst[DIR_TOP_FACE] * backLattice[idx].adj[DIR_TOP_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_RIGHT_EDGE] * backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_FRONT_RIGHT_EDGE] * backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_RIGHT_EDGE] * backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_BACK_EDGE] * backLattice[idx].adj[DIR_TOP_BACK_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_BACK_EDGE] * backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
		macroVelocity /= macroDensity;

		velocities[idx] = macroVelocity;

		float leftTermMiddle = weightMiddle * macroDensity;
		float leftTermAxis = weightAxis * macroDensity;
		float leftTermNonaxial = weightNonaxial * macroDensity;

		float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
		float thirdTerm = 1.5f * macroVelocityDot;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], macroVelocity);
		float firstTerm = 3.0f * dotProd;
		float secondTerm = 4.5f * dotProd * dotProd;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		backLattice[idx].adj[DIR_MIDDLE_VERTEX] -= d_itau * (backLattice[idx].adj[DIR_MIDDLE_VERTEX] - middleEq);
		backLattice[idx].adj[DIR_RIGHT_FACE] -= d_itau * (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
		backLattice[idx].adj[DIR_LEFT_FACE] -= d_itau * (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
		backLattice[idx].adj[DIR_BACK_FACE] -= d_itau * (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
		backLattice[idx].adj[DIR_FRONT_FACE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
		backLattice[idx].adj[DIR_TOP_FACE] -= d_itau * (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
		backLattice[idx].adj[DIR_BOTTOM_FACE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
		backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
		backLattice[idx].adj[DIR_BACK_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
		backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
		backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
		backLattice[idx].adj[DIR_TOP_BACK_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
		backLattice[idx].adj[DIR_TOP_FRONT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
		backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
		backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
		backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
		backLattice[idx].adj[DIR_TOP_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
		backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
		backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


		for (int i = 0; i < 19; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}


}


__global__ void collisionStepKernelShared(Node3D *backLattice, glm::vec3 *velocities) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	__shared__ Node3D cache[256];
	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;


	if (idx < d_latticeSize) {

		cache[cacheIdx] = backLattice[idx];
		//__syncthreads(); // not needed

		float macroDensity = 0.0f;
		for (int i = 0; i < 19; i++) {
			macroDensity += cache[cacheIdx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
		macroVelocity += dirVectorsConst[DIR_LEFT_FACE] * cache[cacheIdx].adj[DIR_LEFT_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_FACE] * cache[cacheIdx].adj[DIR_FRONT_FACE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FACE] * cache[cacheIdx].adj[DIR_BOTTOM_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_LEFT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BACK_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_LEFT_EDGE] * cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FRONT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_FRONT_EDGE] * cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_RIGHT_FACE] * cache[cacheIdx].adj[DIR_RIGHT_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_FACE] * cache[cacheIdx].adj[DIR_BACK_FACE];
		macroVelocity += dirVectorsConst[DIR_TOP_FACE] * cache[cacheIdx].adj[DIR_TOP_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_FRONT_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_BACK_EDGE] * cache[cacheIdx].adj[DIR_TOP_BACK_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_BACK_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE];
		macroVelocity /= macroDensity;

		velocities[idx] = macroVelocity;

		float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
		float leftTermAxis = WEIGHT_AXIS * macroDensity;
		float leftTermNonaxial = WEIGHT_NON_AXIAL * macroDensity;

		float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
		float thirdTerm = 1.5f * macroVelocityDot;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], macroVelocity);
		float firstTerm = 3.0f * dotProd;
		float secondTerm = 4.5f * dotProd * dotProd;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] -= d_itau * (cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] - middleEq);
		cache[cacheIdx].adj[DIR_RIGHT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_RIGHT_FACE] - rightEq);
		cache[cacheIdx].adj[DIR_LEFT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_LEFT_FACE] - leftEq);
		cache[cacheIdx].adj[DIR_BACK_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_FACE] - backEq);
		cache[cacheIdx].adj[DIR_FRONT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_FACE] - frontEq);
		cache[cacheIdx].adj[DIR_TOP_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FACE] - topEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FACE] - bottomEq);
		cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
		cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
		cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
		cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
		cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
		cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
		cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
		cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
		cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
		cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


		for (int i = 0; i < 19; i++) {
			if (cache[cacheIdx].adj[i] < 0.0f) {
				cache[cacheIdx].adj[i] = 0.0f;
			} else if (cache[cacheIdx].adj[i] > 1.0f) {
				cache[cacheIdx].adj[i] = 1.0f;
			}
		}

		backLattice[idx] = cache[cacheIdx];

	}


}

__global__ void updateCollidersKernel(Node3D *backLattice, glm::vec3 *velocities, float *heightMap) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {

		int x = idx % d_latticeWidth;
		int y = (idx / d_latticeWidth) % d_latticeHeight;
		int z = idx / (d_latticeHeight * d_latticeWidth);

		//if (testHM->data[x][z] >= y && testHM->data[x][z] > 0.01f)
		if (heightMap[x + z * d_latticeWidth] >= y && heightMap[x + z * d_latticeWidth] > 0.01f) {

			float right = backLattice[idx].adj[DIR_RIGHT_FACE];
			float left = backLattice[idx].adj[DIR_LEFT_FACE];
			float back = backLattice[idx].adj[DIR_BACK_FACE];
			float front = backLattice[idx].adj[DIR_FRONT_FACE];
			float top = backLattice[idx].adj[DIR_TOP_FACE];
			float bottom = backLattice[idx].adj[DIR_BOTTOM_FACE];
			float backRight = backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
			float backLeft = backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
			float frontRight = backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
			float frontLeft = backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
			float topBack = backLattice[idx].adj[DIR_TOP_BACK_EDGE];
			float topFront = backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
			float bottomBack = backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
			float bottomFront = backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
			float topRight = backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
			float topLeft = backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
			float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
			float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];


			backLattice[idx].adj[DIR_RIGHT_FACE] = left;
			backLattice[idx].adj[DIR_LEFT_FACE] = right;
			backLattice[idx].adj[DIR_BACK_FACE] = front;
			backLattice[idx].adj[DIR_FRONT_FACE] = back;
			backLattice[idx].adj[DIR_TOP_FACE] = bottom;
			backLattice[idx].adj[DIR_BOTTOM_FACE] = top;
			backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = frontLeft;
			backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = frontRight;
			backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = backLeft;
			backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = backRight;
			backLattice[idx].adj[DIR_TOP_BACK_EDGE] = bottomFront;
			backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = bottomBack;
			backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = topFront;
			backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = topBack;
			backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = bottomLeft;
			backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = bottomRight;
			backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = topLeft;
			backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = topRight;

		}
	}
}


__global__ void streamingStepKernel(Node3D *backLattice, Node3D *frontLattice) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {

		int x = idx % d_latticeWidth;
		int y = (idx / d_latticeWidth) % d_latticeHeight;
		int z = idx / (d_latticeHeight * d_latticeWidth);

		backLattice[idx].adj[DIR_MIDDLE_VERTEX] += frontLattice[idx].adj[DIR_MIDDLE_VERTEX];

		int right;
		int left;
		int top;
		int bottom;
		int front;
		int back;

		right = x + 1;
		left = x - 1;
		top = y + 1;
		bottom = y - 1;
		front = z + 1;
		back = z - 1;
		if (right > d_latticeWidth - 1) {
			right = d_latticeWidth - 1;
		}
		if (left < 0) {
			left = 0;
		}
		if (top > d_latticeHeight - 1) {
			top = d_latticeHeight - 1;
		}
		if (bottom < 0) {
			bottom = 0;
		}
		if (front > d_latticeDepth - 1) {
			front = d_latticeDepth - 1;
		}
		if (back < 0) {
			back = 0;
		}

		backLattice[idx].adj[DIR_LEFT_FACE] += frontLattice[getIdxKer(right, y, z)].adj[DIR_LEFT_FACE];
		backLattice[idx].adj[DIR_FRONT_FACE] += frontLattice[getIdxKer(x, y, back)].adj[DIR_FRONT_FACE];
		backLattice[idx].adj[DIR_BOTTOM_FACE] += frontLattice[getIdxKer(x, top, z)].adj[DIR_BOTTOM_FACE];
		backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] += frontLattice[getIdxKer(right, y, back)].adj[DIR_FRONT_LEFT_EDGE];
		backLattice[idx].adj[DIR_BACK_LEFT_EDGE] += frontLattice[getIdxKer(right, y, front)].adj[DIR_BACK_LEFT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] += frontLattice[getIdxKer(right, top, z)].adj[DIR_BOTTOM_LEFT_EDGE];
		backLattice[idx].adj[DIR_TOP_LEFT_EDGE] += frontLattice[getIdxKer(right, bottom, z)].adj[DIR_TOP_LEFT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] += frontLattice[getIdxKer(x, top, back)].adj[DIR_BOTTOM_FRONT_EDGE];
		backLattice[idx].adj[DIR_TOP_FRONT_EDGE] += frontLattice[getIdxKer(x, bottom, back)].adj[DIR_TOP_FRONT_EDGE];
		backLattice[idx].adj[DIR_RIGHT_FACE] += frontLattice[getIdxKer(left, y, z)].adj[DIR_RIGHT_FACE];
		backLattice[idx].adj[DIR_BACK_FACE] += frontLattice[getIdxKer(x, y, front)].adj[DIR_BACK_FACE];
		backLattice[idx].adj[DIR_TOP_FACE] += frontLattice[getIdxKer(x, bottom, z)].adj[DIR_TOP_FACE];
		backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] += frontLattice[getIdxKer(left, y, front)].adj[DIR_BACK_RIGHT_EDGE];
		backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] += frontLattice[getIdxKer(left, y, back)].adj[DIR_FRONT_RIGHT_EDGE];
		backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] += frontLattice[getIdxKer(left, bottom, z)].adj[DIR_TOP_RIGHT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] += frontLattice[getIdxKer(left, top, z)].adj[DIR_BOTTOM_RIGHT_EDGE];
		backLattice[idx].adj[DIR_TOP_BACK_EDGE] += frontLattice[getIdxKer(x, bottom, front)].adj[DIR_TOP_BACK_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] += frontLattice[getIdxKer(x, top, front)].adj[DIR_BOTTOM_BACK_EDGE];

		for (int i = 0; i < 19; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}

}




LBM3D_1D_indices::LBM3D_1D_indices() {
}




LBM3D_1D_indices::LBM3D_1D_indices(glm::vec3 dim, string sceneFilename, float tau, ParticleSystem *particleSystem)
	: LBM(dim, sceneFilename, tau), particleSystem(particleSystem) {

	initScene();


	frontLattice = new Node3D[latticeSize]();
	backLattice = new Node3D[latticeSize]();
	velocities = new glm::vec3[latticeSize]();
	testCollider = new bool[latticeSize]();

	cudaMalloc((void**)&d_frontLattice, sizeof(Node3D) * latticeSize);
	cudaMalloc((void**)&d_backLattice, sizeof(Node3D) * latticeSize);
	cudaMalloc((void**)&d_velocities, sizeof(glm::vec3) * latticeSize);
	cudaMalloc((void**)&d_testCollider, sizeof(bool) * latticeSize);


	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, particleSystem->vbo, cudaGraphicsMapFlagsWriteDiscard);


	cudaMemcpyToSymbol(dirVectorsConst, &directionVectors3D[0], 19 * sizeof(glm::vec3));

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;
	cudaMemcpyToSymbol(WEIGHT_MIDDLE, &weightMiddle, sizeof(float));
	cudaMemcpyToSymbol(WEIGHT_AXIS, &weightAxis, sizeof(float));
	cudaMemcpyToSymbol(WEIGHT_NON_AXIAL, &weightNonaxial, sizeof(float));


	cudaMemcpyToSymbol(d_latticeWidth, &latticeWidth, sizeof(int));
	cudaMemcpyToSymbol(d_latticeHeight, &latticeHeight, sizeof(int));
	cudaMemcpyToSymbol(d_latticeDepth, &latticeDepth, sizeof(int));
	cudaMemcpyToSymbol(d_latticeSize, &latticeSize, sizeof(int));
	cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_itau, &itau, sizeof(float));



	blockDim = dim3(16, 16, 1);
	gridDim = dim3((int)(latticeSize / (16 * 16)) + 1, 1, 1);



	initColliders();

	cudaMemcpy(d_testCollider, testCollider, sizeof(bool) * latticeSize, cudaMemcpyHostToDevice);

	initBuffers();
	initLattice();

	cudaMemcpy(d_backLattice, backLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec3) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);


	//updateInlets(frontLattice);


}


LBM3D_1D_indices::~LBM3D_1D_indices() {

	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;
	delete[] testCollider;

	cudaFree(d_frontLattice);
	cudaFree(d_backLattice);
	cudaFree(d_velocities);
	cudaFree(d_testCollider);

	cudaGraphicsUnregisterResource(cuda_vbo_resource);


}

void LBM3D_1D_indices::initScene() {
	testHM = new HeightMap(sceneFilename, latticeHeight, nullptr);

	latticeWidth = testHM->width;
	latticeDepth = testHM->height;
	latticeSize = latticeWidth * latticeHeight * latticeDepth;

	float *tempHM = new float[latticeWidth * latticeDepth];
	for (int z = 0; z < latticeDepth; z++) {
		for (int x = 0; x < latticeWidth; x++) {
			tempHM[x + z * latticeWidth] = testHM->data[x][z];
		}
	}
	cudaMalloc((void**)&d_heightMap, sizeof(float) * latticeWidth * latticeDepth);
	//cudaMemcpy(d_heightMap, testHM->data, sizeof(float) * latticeWidth * latticeDepth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_heightMap, tempHM, sizeof(float) * latticeWidth * latticeDepth, cudaMemcpyHostToDevice);


	cout << "lattice width = " << latticeWidth << ", height = " << latticeHeight << ", depth = " << latticeDepth << endl;

	delete[] tempHM;

	particleVertices = particleSystem->particleVertices;
	d_numParticles = particleSystem->d_numParticles;

	particleSystem->initParticlePositions(latticeWidth, latticeHeight, latticeDepth);

	//cudaMalloc((void**)&d_particleVertices, sizeof(glm::vec3) * particleSystem->numParticles);


}

void LBM3D_1D_indices::draw(ShaderProgram & shader) {

	//glUseProgram(shader.id);
	//glBindVertexArray(colliderVAO);

	//glPointSize(8.0f);
	//shader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));

	//glDrawArrays(GL_POINTS, 0, colliderVertices.size());


#ifdef DRAW_VELOCITY_ARROWS
	shader.setVec3("color", glm::vec3(0.2f, 0.3f, 1.0f));
	glBindVertexArray(velocityVAO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * velocityArrows.size(), &velocityArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, velocityArrows.size());
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	shader.setVec3("color", glm::vec3(0.8f, 1.0f, 0.6f));

	glBindVertexArray(particleArrowsVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particleArrows.size(), &particleArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, particleArrows.size());
#endif

	testHM->draw();

}





void LBM3D_1D_indices::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();
	collisionStep();
	moveParticles();

	swapLattices();
}

void LBM3D_1D_indices::doStepCUDA() {

	// ============================================= clear back lattice CUDA
	clearBackLatticeKernel << <gridDim, blockDim >> > (d_backLattice);

	// ============================================= update inlets CUDA
	updateInletsKernel << <gridDim, blockDim >> > (d_backLattice, d_velocities);

	// ============================================= streaming step CUDA
	streamingStepKernel << <gridDim, blockDim >> > (d_backLattice, d_frontLattice);

	// ============================================= update colliders CUDA
	updateCollidersKernel << <gridDim, blockDim >> > (d_backLattice, d_velocities, d_heightMap);

	// ============================================= collision step CUDA
	collisionStepKernelShared << <gridDim, blockDim >> > (d_backLattice, d_velocities);

	// ============================================= move particles CUDA - different respawn from CPU !!!

#ifdef USE_INTEROP
	float3 *dptr;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource);
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	moveParticlesKernelInterop << <gridDim, blockDim >> > (dptr, d_velocities, d_numParticles);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
#else // USE_INTEROP - else

	cudaMemcpy(d_particleVertices, particleVertices, sizeof(glm::vec3) * NUM_PARTICLES, cudaMemcpyHostToDevice);
	moveParticlesKernel << <gridDim, blockDim >> > (d_particleVertices, d_velocities);

	//cudaMemcpy(velocities, d_velocities, sizeof(glm::vec3) * latticeSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(particleVertices, d_particleVertices, sizeof(glm::vec3) * NUM_PARTICLES, cudaMemcpyDeviceToHost);


	for (int i = 0; i < NUM_PARTICLES; i++) {
		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1 ||
			particleVertices[i].z <= 0.0f || particleVertices[i].z >= latticeDepth - 1) {

			particleVertices[i] = glm::vec3(0.0f, respawnY, respawnZ++);
			if (respawnZ >= latticeDepth - 1) {
				respawnZ = 0;
				respawnY++;
}
			if (respawnY >= latticeHeight - 1) {
				respawnY = 0;
			}
		}
	}
#endif


	swapLattices();
}








void LBM3D_1D_indices::clearBackLattice() {
	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);
				for (int i = 0; i < 19; i++) {
					backLattice[idx].adj[i] = 0.0f;
				}
			}
			}
		}
#ifdef DRAW_VELOCITY_ARROWS
	velocityArrows.clear();
#endif
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	particleArrows.clear();
#endif

	}

void LBM3D_1D_indices::streamingStep() {


	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);
				backLattice[idx].adj[DIR_MIDDLE_VERTEX] += frontLattice[idx].adj[DIR_MIDDLE_VERTEX];

				int right;
				int left;
				int top;
				int bottom;
				int front;
				int back;

				right = x + 1;
				left = x - 1;
				top = y + 1;
				bottom = y - 1;
				front = z + 1;
				back = z - 1;
				if (right > latticeWidth - 1) {
					right = latticeWidth - 1;
				}
				if (left < 0) {
					left = 0;
				}
				if (top > latticeHeight - 1) {
					top = latticeHeight - 1;
				}
				if (bottom < 0) {
					bottom = 0;
				}
				if (front > latticeDepth - 1) {
					front = latticeDepth - 1;
				}
				if (back < 0) {
					back = 0;
				}

				backLattice[idx].adj[DIR_LEFT_FACE] += frontLattice[getIdx(right, y, z)].adj[DIR_LEFT_FACE];
				backLattice[idx].adj[DIR_FRONT_FACE] += frontLattice[getIdx(x, y, back)].adj[DIR_FRONT_FACE];
				backLattice[idx].adj[DIR_BOTTOM_FACE] += frontLattice[getIdx(x, top, z)].adj[DIR_BOTTOM_FACE];
				backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] += frontLattice[getIdx(right, y, back)].adj[DIR_FRONT_LEFT_EDGE];
				backLattice[idx].adj[DIR_BACK_LEFT_EDGE] += frontLattice[getIdx(right, y, front)].adj[DIR_BACK_LEFT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] += frontLattice[getIdx(right, top, z)].adj[DIR_BOTTOM_LEFT_EDGE];
				backLattice[idx].adj[DIR_TOP_LEFT_EDGE] += frontLattice[getIdx(right, bottom, z)].adj[DIR_TOP_LEFT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] += frontLattice[getIdx(x, top, back)].adj[DIR_BOTTOM_FRONT_EDGE];
				backLattice[idx].adj[DIR_TOP_FRONT_EDGE] += frontLattice[getIdx(x, bottom, back)].adj[DIR_TOP_FRONT_EDGE];
				backLattice[idx].adj[DIR_RIGHT_FACE] += frontLattice[getIdx(left, y, z)].adj[DIR_RIGHT_FACE];
				backLattice[idx].adj[DIR_BACK_FACE] += frontLattice[getIdx(x, y, front)].adj[DIR_BACK_FACE];
				backLattice[idx].adj[DIR_TOP_FACE] += frontLattice[getIdx(x, bottom, z)].adj[DIR_TOP_FACE];
				backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] += frontLattice[getIdx(left, y, front)].adj[DIR_BACK_RIGHT_EDGE];
				backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] += frontLattice[getIdx(left, y, back)].adj[DIR_FRONT_RIGHT_EDGE];
				backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] += frontLattice[getIdx(left, bottom, z)].adj[DIR_TOP_RIGHT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] += frontLattice[getIdx(left, top, z)].adj[DIR_BOTTOM_RIGHT_EDGE];
				backLattice[idx].adj[DIR_TOP_BACK_EDGE] += frontLattice[getIdx(x, bottom, front)].adj[DIR_TOP_BACK_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] += frontLattice[getIdx(x, top, front)].adj[DIR_BOTTOM_BACK_EDGE];

				for (int i = 0; i < 19; i++) {
					if (backLattice[idx].adj[i] < 0.0f) {
						backLattice[idx].adj[i] = 0.0f;
					} else if (backLattice[idx].adj[i] > 1.0f) {
						backLattice[idx].adj[i] = 1.0f;
					}
				}
			}
		}
	}

}

void LBM3D_1D_indices::collisionStep() {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;

	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {

				int idx = getIdx(x, y, z);

				float macroDensity = calculateMacroscopicDensity(x, y, z);
				glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, z, macroDensity);

				velocities[idx] = macroVelocity;

#ifdef DRAW_VELOCITY_ARROWS
				velocityArrows.push_back(glm::vec3(x, y, z));
				velocityArrows.push_back(glm::vec3(x, y, z) + velocities[idx] * 2.0f);
#endif


				float leftTermMiddle = weightMiddle * macroDensity;
				float leftTermAxis = weightAxis * macroDensity;
				float leftTermNonaxial = weightNonaxial * macroDensity;

				float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
				float thirdTerm = 1.5f * macroVelocityDot;

				float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

				float dotProd = glm::dot(vRight, macroVelocity);
				float firstTerm = 3.0f * dotProd;
				float secondTerm = 4.5f * dotProd * dotProd;
				float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTop, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottom, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBackRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBackLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vFrontRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vFrontLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottomBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vBottomFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottomRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vBottomLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				backLattice[idx].adj[DIR_MIDDLE_VERTEX] -= itau * (backLattice[idx].adj[DIR_MIDDLE_VERTEX] - middleEq);
				backLattice[idx].adj[DIR_RIGHT_FACE] -= itau * (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
				backLattice[idx].adj[DIR_LEFT_FACE] -= itau * (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
				backLattice[idx].adj[DIR_BACK_FACE] -= itau * (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
				backLattice[idx].adj[DIR_FRONT_FACE] -= itau * (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
				backLattice[idx].adj[DIR_TOP_FACE] -= itau * (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
				backLattice[idx].adj[DIR_BOTTOM_FACE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
				backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
				backLattice[idx].adj[DIR_BACK_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
				backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
				backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
				backLattice[idx].adj[DIR_TOP_BACK_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
				backLattice[idx].adj[DIR_TOP_FRONT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
				backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
				backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
				backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
				backLattice[idx].adj[DIR_TOP_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
				backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
				backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


				for (int i = 0; i < 19; i++) {
					if (backLattice[idx].adj[i] < 0.0f) {
						backLattice[idx].adj[i] = 0.0f;
					} else if (backLattice[idx].adj[i] > 1.0f) {
						backLattice[idx].adj[i] = 1.0f;
					}
				}





			}
		}
	}

}


void LBM3D_1D_indices::moveParticles() {

	glm::vec3 adjVelocities[8];
	for (int i = 0; i < particleSystem->numParticles; i++) {

		float x = particleVertices[i].x;
		float y = particleVertices[i].y;
		float z = particleVertices[i].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[getIdx(leftX, topY, backZ)];
		adjVelocities[1] = velocities[getIdx(rightX, topY, backZ)];
		adjVelocities[2] = velocities[getIdx(leftX, bottomY, backZ)];
		adjVelocities[3] = velocities[getIdx(rightX, bottomY, backZ)];
		adjVelocities[4] = velocities[getIdx(leftX, topY, frontZ)];
		adjVelocities[5] = velocities[getIdx(rightX, topY, frontZ)];
		adjVelocities[6] = velocities[getIdx(leftX, bottomY, frontZ)];
		adjVelocities[7] = velocities[getIdx(rightX, bottomY, frontZ)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;
		float depthRatio = z - backZ;

		glm::vec3 topBackVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec3 bottomBackVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec3 backVelocity = bottomBackVelocity * verticalRatio + topBackVelocity * (1.0f - verticalRatio);

		glm::vec3 topFrontVelocity = adjVelocities[4] * horizontalRatio + adjVelocities[5] * (1.0f - horizontalRatio);
		glm::vec3 bottomFrontVelocity = adjVelocities[6] * horizontalRatio + adjVelocities[7] * (1.0f - horizontalRatio);

		glm::vec3 frontVelocity = bottomFrontVelocity * verticalRatio + topFrontVelocity * (1.0f - verticalRatio);

		glm::vec3 finalVelocity = backVelocity * depthRatio + frontVelocity * (1.0f - depthRatio);

#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		particleArrows.push_back(particleVertices[i]);
#endif
		particleVertices[i] += finalVelocity;
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		glm::vec3 tmp = particleVertices[i] + 10.0f * finalVelocity;
		particleArrows.push_back(tmp);
#endif

		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1 ||
			particleVertices[i].z <= 0.0f || particleVertices[i].z >= latticeDepth - 1) {

			particleVertices[i] = glm::vec3(0.0f, respawnY, respawnZ++);
			if (respawnZ >= latticeDepth - 1) {
				respawnZ = 0;
				respawnY++;
			}
			if (respawnY >= latticeHeight - 1) {
				respawnY = 0;
			}
		}

	}
}

void LBM3D_1D_indices::updateInlets() {

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;


	float macroDensity = 1.0f;
	glm::vec3 macroVelocity = glm::vec3(0.4f, 0.0f, 0.0f);


	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermNonaxial = weightNonaxial * macroDensity;

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	for (int z = 0; z < latticeDepth; z++) {
		for (int y = 0; y < latticeHeight; y++) {

			int idx = getIdx(0, y, z);

			backLattice[idx].adj[DIR_MIDDLE_VERTEX] = middleEq;
			backLattice[idx].adj[DIR_RIGHT_FACE] = rightEq;
			backLattice[idx].adj[DIR_LEFT_FACE] = leftEq;
			backLattice[idx].adj[DIR_BACK_FACE] = backEq;
			backLattice[idx].adj[DIR_FRONT_FACE] = frontEq;
			backLattice[idx].adj[DIR_TOP_FACE] = topEq;
			backLattice[idx].adj[DIR_BOTTOM_FACE] = bottomEq;
			backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = backRightEq;
			backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = backLeftEq;
			backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = frontRightEq;
			backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = frontLeftEq;
			backLattice[idx].adj[DIR_TOP_BACK_EDGE] = topBackEq;
			backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = topFrontEq;
			backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = bottomBackEq;
			backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = bottomFrontEq;
			backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = topRightEq;
			backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = topLeftEq;
			backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = bottomRightEq;
			backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = bottomLeftEq;


			for (int i = 0; i < 19; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}
		}
	}
}


void LBM3D_1D_indices::updateInlets(Node3D *lattice) {

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;


	float macroDensity = 1.0f;
	glm::vec3 macroVelocity = glm::vec3(1.0f, 0.0f, 0.0f);


	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermNonaxial = weightNonaxial * macroDensity;

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	for (int z = 0; z < latticeDepth; z++) {
		for (int y = 0; y < latticeHeight; y++) {

			int idx = getIdx(0, y, z);

			lattice[idx].adj[DIR_MIDDLE_VERTEX] = middleEq;
			lattice[idx].adj[DIR_RIGHT_FACE] = rightEq;
			lattice[idx].adj[DIR_LEFT_FACE] = leftEq;
			lattice[idx].adj[DIR_BACK_FACE] = backEq;
			lattice[idx].adj[DIR_FRONT_FACE] = frontEq;
			lattice[idx].adj[DIR_TOP_FACE] = topEq;
			lattice[idx].adj[DIR_BOTTOM_FACE] = bottomEq;
			lattice[idx].adj[DIR_BACK_RIGHT_EDGE] = backRightEq;
			lattice[idx].adj[DIR_BACK_LEFT_EDGE] = backLeftEq;
			lattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = frontRightEq;
			lattice[idx].adj[DIR_FRONT_LEFT_EDGE] = frontLeftEq;
			lattice[idx].adj[DIR_TOP_BACK_EDGE] = topBackEq;
			lattice[idx].adj[DIR_TOP_FRONT_EDGE] = topFrontEq;
			lattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = bottomBackEq;
			lattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = bottomFrontEq;
			lattice[idx].adj[DIR_TOP_RIGHT_EDGE] = topRightEq;
			lattice[idx].adj[DIR_TOP_LEFT_EDGE] = topLeftEq;
			lattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = bottomRightEq;
			lattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = bottomLeftEq;


			for (int i = 0; i < 19; i++) {
				if (lattice[idx].adj[i] < 0.0f) {
					lattice[idx].adj[i] = 0.0f;
				} else if (lattice[idx].adj[i] > 1.0f) {
					lattice[idx].adj[i] = 1.0f;
				}
			}
		}
	}
}



void LBM3D_1D_indices::updateColliders() {


	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);

				//continue; //////////////////////////////////////////////////////////////////////////////////////////
				if (/*z == 0 || z == latticeDepth - 1 ||
					y == 0 || y == latticeHeight - 1 ||*/
					/*testCollider[idx]*/
					(testHM->data[x][z] >= y && testHM->data[x][z] > 0.01f) ||
					y == 0
					) {



					float right = backLattice[idx].adj[DIR_RIGHT_FACE];
					float left = backLattice[idx].adj[DIR_LEFT_FACE];
					float back = backLattice[idx].adj[DIR_BACK_FACE];
					float front = backLattice[idx].adj[DIR_FRONT_FACE];
					float top = backLattice[idx].adj[DIR_TOP_FACE];
					float bottom = backLattice[idx].adj[DIR_BOTTOM_FACE];
					float backRight = backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
					float backLeft = backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
					float frontRight = backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
					float frontLeft = backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
					float topBack = backLattice[idx].adj[DIR_TOP_BACK_EDGE];
					float topFront = backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
					float bottomBack = backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
					float bottomFront = backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
					float topRight = backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
					float topLeft = backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
					float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
					float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];


					backLattice[idx].adj[DIR_RIGHT_FACE] = left;
					backLattice[idx].adj[DIR_LEFT_FACE] = right;
					backLattice[idx].adj[DIR_BACK_FACE] = front;
					backLattice[idx].adj[DIR_FRONT_FACE] = back;
					backLattice[idx].adj[DIR_TOP_FACE] = bottom;
					backLattice[idx].adj[DIR_BOTTOM_FACE] = top;
					backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = frontLeft;
					backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = frontRight;
					backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = backLeft;
					backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = backRight;
					backLattice[idx].adj[DIR_TOP_BACK_EDGE] = bottomFront;
					backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = bottomBack;
					backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = topFront;
					backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = topBack;
					backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = bottomLeft;
					backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = bottomRight;
					backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = topLeft;
					backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = topRight;
					/*
										float macroDensity = calculateMacroscopicDensity(x, y, z);
										glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, z, macroDensity);
										velocities[idx] = macroVelocity;
					*/
				}



			}
		}


	}



}

void LBM3D_1D_indices::initBuffers() {

	glGenVertexArrays(1, &colliderVAO);
	glBindVertexArray(colliderVAO);
	glGenBuffers(1, &colliderVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colliderVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * colliderVertices.size(), &colliderVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);


#ifdef DRAW_VELOCITY_ARROWS
	// Velocity arrows
	glGenVertexArrays(1, &velocityVAO);
	glBindVertexArray(velocityVAO);
	glGenBuffers(1, &velocityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	// Particle arrows
	glGenVertexArrays(1, &particleArrowsVAO);
	glBindVertexArray(particleArrowsVAO);
	glGenBuffers(1, &particleArrowsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
#endif


	glBindVertexArray(0);


}

void LBM3D_1D_indices::initLattice() {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;
	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);
				frontLattice[idx].adj[DIR_MIDDLE_VERTEX] = weightMiddle;
				for (int i = 1; i <= 6; i++) {
					frontLattice[idx].adj[i] = weightAxis;
				}
				for (int i = 7; i <= 18; i++) {
					frontLattice[idx].adj[i] = weightNonaxial;
				}
			}
		}
	}


}

void LBM3D_1D_indices::initColliders() {

	// test sphere collider
	//glm::vec3 center(latticeWidth / 2.0f, latticeHeight / 2.0f, latticeDepth / 2.0f);
	//float radius = latticeDepth / 2.0f;

	//for (int x = 0; x < latticeWidth; x++) {
	//	for (int y = 0; y < latticeHeight; y++) {
	//		for (int z = 0; z < latticeDepth; z++) {

	//			if (glm::distance(center, glm::vec3(x, y, z)) <= radius) {
	//				testCollider[x][y][z] = true;
	//			}

	//		}
	//	}
	//}


	for (int x = latticeWidth / 3.0f; x < latticeWidth / 2.0f; x++) {
		for (int y = latticeHeight / 4.0f; y < latticeHeight / 3.0f; y++) {
			for (int z = latticeDepth / 3.0f; z < latticeDepth / 2.0f; z++) {
				//cout << " =============================== " << endl;
				//int idx = getIdx(x, y, z);
				//cout << x << ", " << y << ", " << z << endl;
				//cout << idx << endl;

				//int xDirection = idx % latticeHeight;
				//int yDirection = (idx / latticeHeight) % latticeWidth;
				//int zDirection = idx / (latticeWidth * latticeHeight);
				//cout << xDirection << ", " << yDirection << ", " << zDirection << endl;


				testCollider[getIdx(x, y, z)] = true;
				colliderVertices.push_back(glm::vec3(x, y, z));
			}
		}
	}




}


void LBM3D_1D_indices::swapLattices() {
	Node3D *tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;
	tmp = d_frontLattice;
	d_frontLattice = d_backLattice;
	d_backLattice = tmp;
}

float LBM3D_1D_indices::calculateMacroscopicDensity(int x, int y, int z) {
	float macroDensity = 0.0f;
	int idx = getIdx(x, y, z);
	for (int i = 0; i < 19; i++) {
		macroDensity += backLattice[idx].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM3D_1D_indices::calculateMacroscopicVelocity(int x, int y, int z, float macroDensity) {

	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	int idx = getIdx(x, y, z);
	//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
	macroVelocity += vLeft * backLattice[idx].adj[DIR_LEFT_FACE];
	macroVelocity += vFront * backLattice[idx].adj[DIR_FRONT_FACE];
	macroVelocity += vBottom * backLattice[idx].adj[DIR_BOTTOM_FACE];
	macroVelocity += vFrontLeft * backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
	macroVelocity += vBackLeft * backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
	macroVelocity += vBottomLeft * backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];
	macroVelocity += vTopLeft * backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
	macroVelocity += vBottomFront * backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
	macroVelocity += vTopFront * backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
	macroVelocity += vRight * backLattice[idx].adj[DIR_RIGHT_FACE];
	macroVelocity += vBack * backLattice[idx].adj[DIR_BACK_FACE];
	macroVelocity += vTop * backLattice[idx].adj[DIR_TOP_FACE];
	macroVelocity += vBackRight * backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
	macroVelocity += vFrontRight * backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
	macroVelocity += vTopRight * backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
	macroVelocity += vBottomRight * backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
	macroVelocity += vTopBack * backLattice[idx].adj[DIR_TOP_BACK_EDGE];
	macroVelocity += vBottomBack * backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
	macroVelocity /= macroDensity;

	return macroVelocity;
}
