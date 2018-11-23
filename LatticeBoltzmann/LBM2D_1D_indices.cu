#include "LBM2D_1D_indices.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>


//#define BLOCK_DIM 512

__device__ int getIdxKernel(int x, int y) {
	return x + y * GRID_WIDTH;
}



__global__ void moveParticlesKernel(glm::vec3 *particleVertices, glm::vec2 *velocities) {


	glm::vec2 adjVelocities[4];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	while (idx < NUM_PARTICLES) {
		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;


		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[getIdxKernel(leftX, topY)];
		adjVelocities[1] = velocities[getIdxKernel(rightX, topY)];
		adjVelocities[2] = velocities[getIdxKernel(leftX, bottomY)];
		adjVelocities[3] = velocities[getIdxKernel(rightX, bottomY)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);

		particleVertices[idx] += glm::vec3(finalVelocity, 0.0f);
	
		idx += blockDim.x * gridDim.x;


	}
}

__global__ void moveParticlesKernelInterop(float3 *particleVertices, glm::vec2 *velocities) {


	glm::vec2 adjVelocities[4];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	while (idx < NUM_PARTICLES) {
		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;


		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[getIdxKernel(leftX, topY)];
		adjVelocities[1] = velocities[getIdxKernel(rightX, topY)];
		adjVelocities[2] = velocities[getIdxKernel(leftX, bottomY)];
		adjVelocities[3] = velocities[getIdxKernel(rightX, bottomY)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);

		//particleVertices[idx] += make_float3(finalVelocity.x, 0.0f);
		particleVertices[idx].x += finalVelocity.x;
		particleVertices[idx].y += finalVelocity.y;


		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= GRID_WIDTH - 1 ||
			particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= GRID_HEIGHT - 1) {
#ifdef MIRROR_SIDES
			if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= GRID_WIDTH - 1) {
				//particleVertices[idx] = glm::vec3(0, respawnIdx++, 0.0f);
				particleVertices[idx].x = 0.0f;
				particleVertices[idx].z = 0.0f;
			} else {
				//particleVertices[i] = glm::vec3(particleVertices[i].x, (int)(particleVertices[i].y + GRID_HEIGHT - 1) % (GRID_HEIGHT - 1), 0.0f);
				particleVertices[idx].y = (float)((int)(particleVertices[idx].y + GRID_HEIGHT - 1) % (GRID_HEIGHT - 1));
				particleVertices[idx].z = 0.0f;
			}
#else
			//particleVertices[i] = glm::vec3(0, respawnIdx++, 0.0f);
			particleVertices[idx].x = 0.0f;
			particleVertices[idx].z = 0.0f;
#endif
		}
		
		idx += blockDim.x * gridDim.x;


	}
}



__global__ void clearBackLatticeKernel(Node *backLattice) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < GRID_WIDTH * GRID_HEIGHT) {
		for (int i = 0; i < 9; i++) {
			backLattice[idx].adj[i] = 0.0f;
		}
	}
}

__global__ void streamingStepKernel(Node *backLattice, Node *frontLattice) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < GRID_WIDTH * GRID_HEIGHT) {

		int x = idx % GRID_WIDTH;
		int y = (idx / GRID_WIDTH) % GRID_HEIGHT;

		backLattice[idx].adj[DIR_MIDDLE] += frontLattice[idx].adj[DIR_MIDDLE];

		int right;
		int left;
		int top;
		int bottom;

		right = x + 1;
		left = x - 1;
		top = y + 1;
		bottom = y - 1;
		if (right > GRID_WIDTH - 1) {
			right = GRID_WIDTH - 1;
		}
		if (left < 0) {
			left = 0;
		}
		if (top > GRID_HEIGHT - 1) {
			top = GRID_HEIGHT - 1;
		}
		if (bottom < 0) {
			bottom = 0;
		}


		backLattice[idx].adj[DIR_RIGHT] += frontLattice[getIdxKernel(left, y)].adj[DIR_RIGHT];
		backLattice[idx].adj[DIR_TOP] += frontLattice[getIdxKernel(x, bottom)].adj[DIR_TOP];
		backLattice[idx].adj[DIR_LEFT] += frontLattice[getIdxKernel(right, y)].adj[DIR_LEFT];
		backLattice[idx].adj[DIR_BOTTOM] += frontLattice[getIdxKernel(x, top)].adj[DIR_BOTTOM];
		backLattice[idx].adj[DIR_TOP_RIGHT] += frontLattice[getIdxKernel(left, bottom)].adj[DIR_TOP_RIGHT];
		backLattice[idx].adj[DIR_TOP_LEFT] += frontLattice[getIdxKernel(right, bottom)].adj[DIR_TOP_LEFT];
		backLattice[idx].adj[DIR_BOTTOM_LEFT] += frontLattice[getIdxKernel(right, top)].adj[DIR_BOTTOM_LEFT];
		backLattice[idx].adj[DIR_BOTTOM_RIGHT] += frontLattice[getIdxKernel(left, top)].adj[DIR_BOTTOM_RIGHT];

		for (int i = 0; i < 9; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}


}

__global__ void updateInletsKernel(Node *lattice) {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = glm::vec3(1.0f, 0.0f, 0.0f);

	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

	// let's find the equilibrium
	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermDiagonal = weightDiagonal * macroDensity;

	// optimize these operations later

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot / LAT_SPEED_SQ;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	// this can all be rewritten into arrays + for cycles!
	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd / LAT_SPEED;
	float secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int x = idx % GRID_WIDTH;

	if (x == 0 && idx < GRID_WIDTH * GRID_HEIGHT) {

		lattice[idx].adj[DIR_MIDDLE] = middleEq;
		lattice[idx].adj[DIR_RIGHT] = rightEq;
		lattice[idx].adj[DIR_TOP] = topEq;
		lattice[idx].adj[DIR_LEFT] = leftEq;
		lattice[idx].adj[DIR_BOTTOM] = bottomEq;
		lattice[idx].adj[DIR_TOP_RIGHT] = topRightEq;
		lattice[idx].adj[DIR_TOP_LEFT] = topLeftEq;
		lattice[idx].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		lattice[idx].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (lattice[idx].adj[i] < 0.0f) {
				lattice[idx].adj[i] = 0.0f;
			} else if (lattice[idx].adj[i] > 1.0f) {
				lattice[idx].adj[i] = 1.0f;
			}
		}
	}

}

__global__ void updateCollidersKernel(Node *backLattice, bool *tCol) {


	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < GRID_WIDTH * GRID_HEIGHT) {
		if (tCol[idx]) {

			float right = backLattice[idx].adj[DIR_RIGHT];
			float top = backLattice[idx].adj[DIR_TOP];
			float left = backLattice[idx].adj[DIR_LEFT];
			float bottom = backLattice[idx].adj[DIR_BOTTOM];
			float topRight = backLattice[idx].adj[DIR_TOP_RIGHT];
			float topLeft = backLattice[idx].adj[DIR_TOP_LEFT];
			float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT];
			float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT];
			backLattice[idx].adj[DIR_RIGHT] = left;
			backLattice[idx].adj[DIR_TOP] = bottom;
			backLattice[idx].adj[DIR_LEFT] = right;
			backLattice[idx].adj[DIR_BOTTOM] = top;
			backLattice[idx].adj[DIR_TOP_RIGHT] = bottomLeft;
			backLattice[idx].adj[DIR_TOP_LEFT] = bottomRight;
			backLattice[idx].adj[DIR_BOTTOM_LEFT] = topRight;
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] = topLeft;
		}
	}
}


__global__ void collisionStepKernel(Node *backLattice, glm::vec2 *velocities) {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1D array kernel
	int cacheIdx = threadIdx.x;

	__shared__ Node cache[BLOCK_DIM];


	if (idx < GRID_WIDTH * GRID_HEIGHT) {

		cache[cacheIdx] = backLattice[idx];


		float macroDensity = 0.0f;
		for (int i = 0; i < 9; i++) {
			macroDensity += cache[cacheIdx].adj[i];
		}

		const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
		const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
		const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
		const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
		const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
		const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
		const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
		const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		macroVelocity += LAT_SPEED * vRight * cache[cacheIdx].adj[DIR_RIGHT];
		macroVelocity += LAT_SPEED * vTop * cache[cacheIdx].adj[DIR_TOP];
		macroVelocity += LAT_SPEED * vLeft * cache[cacheIdx].adj[DIR_LEFT];
		macroVelocity += LAT_SPEED * vBottom * cache[cacheIdx].adj[DIR_BOTTOM];
		macroVelocity += LAT_SPEED * vTopRight * cache[cacheIdx].adj[DIR_TOP_RIGHT];
		macroVelocity += LAT_SPEED * vTopLeft * cache[cacheIdx].adj[DIR_TOP_LEFT];
		macroVelocity += LAT_SPEED * vBottomLeft * cache[cacheIdx].adj[DIR_BOTTOM_LEFT];
		macroVelocity += LAT_SPEED * vBottomRight * cache[cacheIdx].adj[DIR_BOTTOM_RIGHT];
		macroVelocity /= macroDensity;


		velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);



		// let's find the equilibrium
		float leftTermMiddle = weightMiddle * macroDensity;
		float leftTermAxis = weightAxis * macroDensity;
		float leftTermDiagonal = weightDiagonal * macroDensity;

		// optimize these operations later

		float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
		float thirdTerm = 1.5f * macroVelocityDot / LAT_SPEED_SQ;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		// this can all be rewritten into arrays + for cycles!
		float dotProd = glm::dot(vRight, macroVelocity);
		float firstTerm = 3.0f * dotProd / LAT_SPEED;
		float secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(vTop, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(vLeft, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(vBottom, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(vTopRight, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(vTopLeft, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(vBottomLeft, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(vBottomRight, macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);

		cache[cacheIdx].adj[DIR_MIDDLE] -= ITAU * (cache[cacheIdx].adj[DIR_MIDDLE] - middleEq);
		cache[cacheIdx].adj[DIR_RIGHT] -= ITAU * (cache[cacheIdx].adj[DIR_RIGHT] - rightEq);
		cache[cacheIdx].adj[DIR_TOP] -= ITAU * (cache[cacheIdx].adj[DIR_TOP] - topEq);
		cache[cacheIdx].adj[DIR_LEFT] -= ITAU * (cache[cacheIdx].adj[DIR_LEFT] - leftEq);
		cache[cacheIdx].adj[DIR_BOTTOM] -= ITAU * (cache[cacheIdx].adj[DIR_BOTTOM] - bottomEq);
		cache[cacheIdx].adj[DIR_TOP_RIGHT] -= ITAU * (cache[cacheIdx].adj[DIR_TOP_RIGHT] - topRightEq);
		cache[cacheIdx].adj[DIR_TOP_LEFT] -= ITAU * (cache[cacheIdx].adj[DIR_TOP_LEFT] - topLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_LEFT] -= ITAU * (cache[cacheIdx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_RIGHT] -= ITAU * (cache[cacheIdx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


		for (int i = 0; i < 9; i++) {
			if (cache[cacheIdx].adj[i] < 0.0f) {
				cache[cacheIdx].adj[i] = 0.0f;
			} else if (cache[cacheIdx].adj[i] > 1.0f) {
				cache[cacheIdx].adj[i] = 1.0f;
			}
		}

		backLattice[idx] = cache[cacheIdx];

	}
}



void LBM2D_1D_indices::collisionStepCUDA() {

	cudaMemcpy(d_backLattice, backLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec2) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);

	collisionStepKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice, d_velocities);

	//cudaDeviceSynchronize();
	cudaMemcpy(backLattice, d_backLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities, d_velocities, sizeof(glm::vec2) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

}


LBM2D_1D_indices::LBM2D_1D_indices() {
}

LBM2D_1D_indices::LBM2D_1D_indices(ParticleSystem *particleSystem) : particleSystem(particleSystem) {

	particleVertices = particleSystem->particleVertices;
	frontLattice = new Node[GRID_WIDTH * GRID_HEIGHT]();
	backLattice = new Node[GRID_WIDTH * GRID_HEIGHT]();
	velocities = new glm::vec2[GRID_WIDTH * GRID_HEIGHT]();

	//cudaMallocPitch((void**)&d_frontLattice, &frontLatticePitch, GRID_WIDTH, GRID_HEIGHT);
	cudaMalloc((void**)&d_frontLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT);
	cudaMalloc((void**)&d_backLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT);
	cudaMalloc((void**)&d_velocities, sizeof(glm::vec2) * GRID_WIDTH * GRID_HEIGHT);
	cudaMalloc((void**)&d_tCol, sizeof(bool) * GRID_WIDTH * GRID_HEIGHT);
	cudaMalloc((void**)&d_particleVertices, sizeof(glm::vec3) * NUM_PARTICLES);

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, particleSystem->vbo, cudaGraphicsMapFlagsWriteDiscard);

	initTestCollider();

	initBuffers();
	initLattice();
	//updateInlets(frontLattice);

#ifdef USE_CUDA
	cudaMemcpy(d_backLattice, backLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec2) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);
#endif

}


LBM2D_1D_indices::~LBM2D_1D_indices() {
	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;

	delete tCol;

	cudaFree(d_frontLattice);
	cudaFree(d_backLattice);
	cudaFree(d_particleVertices);
	cudaFree(d_tCol);
	cudaFree(d_velocities);

	cudaGraphicsUnregisterResource(cuda_vbo_resource);

}

void LBM2D_1D_indices::draw(ShaderProgram &shader) {
	glPointSize(0.4f);
	shader.setVec3("color", glm::vec3(0.4f, 0.4f, 0.1f));
	glUseProgram(shader.id);

	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, GRID_WIDTH * GRID_HEIGHT);


	//cout << "Velocity arrows size = " << velocityArrows.size() << endl;

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

	// Draw test collider
	tCol->draw(shader);





}

void LBM2D_1D_indices::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();
	//updateCollidersAlt();
	collisionStep();
	//collisionStepStreamlined();

	moveParticles();

	swapLattices();


}

void LBM2D_1D_indices::doStepCUDA() {


	// ============================================= clear back lattice CUDA
	clearBackLatticeKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice);

	// ============================================= update inlets CUDA
	updateInletsKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice);

	// ============================================= streaming step CUDA
	streamingStepKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice, d_frontLattice);

	// ============================================= update colliders CUDA
	updateCollidersKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice, d_tCol);

	// ============================================= collision step CUDA
	collisionStepKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_backLattice, d_velocities);

	// ============================================= move particles CUDA - different respawn from CPU !!!

#ifdef USE_INTEROP
	float3 *dptr;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource);
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	moveParticlesKernelInterop << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (dptr, d_velocities);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
#else // USE_INTEROP - else

	//moveParticles();


	cudaMemcpy(d_particleVertices, particleVertices, sizeof(glm::vec3) * NUM_PARTICLES, cudaMemcpyHostToDevice);

	moveParticlesKernel << <(int)((GRID_WIDTH * GRID_HEIGHT) / BLOCK_DIM) + 1, BLOCK_DIM >> > (d_particleVertices, d_velocities);

	cudaMemcpy(particleVertices, d_particleVertices, sizeof(glm::vec3) * NUM_PARTICLES, cudaMemcpyDeviceToHost);

	for (int i = 0; i < NUM_PARTICLES; i++) {

		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= GRID_HEIGHT - 1) {
#ifdef MIRROR_SIDES
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1) {
				particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
				if (respawnIndex >= GRID_HEIGHT - 1) {
					respawnIndex = 0;
				}
			} else {
				particleVertices[i] = glm::vec3(particleVertices[i].x, (int)(particleVertices[i].y + GRID_HEIGHT - 1) % (GRID_HEIGHT - 1), 0.0f);
			}
#else
			particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
			if (respawnIndex >= GRID_HEIGHT - 1) {
				respawnIndex = 0;
			}
#endif
		}
	}

#endif // USE_INTEROP


	swapLattices();
}

void LBM2D_1D_indices::clearBackLattice() {
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int i = 0; i < 9; i++) {
				backLattice[getIdx(x, y)].adj[i] = 0.0f;
			}
		}
	}
	velocityArrows.clear();
	particleArrows.clear();
}

void LBM2D_1D_indices::streamingStep() {

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			backLattice[getIdx(x, y)].adj[DIR_MIDDLE] += frontLattice[getIdx(x, y)].adj[DIR_MIDDLE];

			int right;
			int left;
			int top;
			int bottom;

			right = x + 1;
			left = x - 1;
			top = y + 1;
			bottom = y - 1;
			if (right > GRID_WIDTH - 1) {
				right = GRID_WIDTH - 1;
			}
			if (left < 0) {
				left = 0;
			}
			if (top > GRID_HEIGHT - 1) {
				top = GRID_HEIGHT - 1;
			}
			if (bottom < 0) {
				bottom = 0;
			}


			backLattice[getIdx(x, y)].adj[DIR_RIGHT] += frontLattice[getIdx(left, y)].adj[DIR_RIGHT];
			backLattice[getIdx(x, y)].adj[DIR_TOP] += frontLattice[getIdx(x, bottom)].adj[DIR_TOP];
			backLattice[getIdx(x, y)].adj[DIR_LEFT] += frontLattice[getIdx(right, y)].adj[DIR_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM] += frontLattice[getIdx(x, top)].adj[DIR_BOTTOM];
			backLattice[getIdx(x, y)].adj[DIR_TOP_RIGHT] += frontLattice[getIdx(left, bottom)].adj[DIR_TOP_RIGHT];
			backLattice[getIdx(x, y)].adj[DIR_TOP_LEFT] += frontLattice[getIdx(right, bottom)].adj[DIR_TOP_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM_LEFT] += frontLattice[getIdx(right, top)].adj[DIR_BOTTOM_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM_RIGHT] += frontLattice[getIdx(left, top)].adj[DIR_BOTTOM_RIGHT];

			for (int i = 0; i < 9; i++) {
				if (backLattice[getIdx(x, y)].adj[i] < 0.0f) {
					backLattice[getIdx(x, y)].adj[i] = 0.0f;
				} else if (backLattice[getIdx(x, y)].adj[i] > 1.0f) {
					backLattice[getIdx(x, y)].adj[i] = 1.0f;
				}
			}

		}
	}
}

void LBM2D_1D_indices::collisionStep() {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			/*if (x == 0 || x == GRID_WIDTH - 1) {
				continue;
			}
			if (y == 0 || y == GRID_HEIGHT - 1) {
				continue;
			}*/


			float macroDensity = calculateMacroscopicDensity(x, y);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);

			int idx = getIdx(x, y);
			velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);


			velocityArrows.push_back(glm::vec3(x, y, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[idx] * 5.0f, -1.0f) + glm::vec3(x, y, 0.0f));



			// let's find the equilibrium
			float leftTermMiddle = weightMiddle * macroDensity;
			float leftTermAxis = weightAxis * macroDensity;
			float leftTermDiagonal = weightDiagonal * macroDensity;

			// optimize these operations later

			float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
			float thirdTerm = 1.5f * macroVelocityDot;

			float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

			// this can all be rewritten into arrays + for cycles!
			float dotProd = glm::dot(vRight, macroVelocity);
			float firstTerm = 3.0f * dotProd;
			float secondTerm = 4.5f * dotProd * dotProd;
			float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

			dotProd = glm::dot(vTop, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

			dotProd = glm::dot(vLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottom, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vTopRight, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vTopLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottomLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottomRight, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);

			backLattice[idx].adj[DIR_MIDDLE] -= ITAU * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);
			backLattice[idx].adj[DIR_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_RIGHT] - rightEq);
			backLattice[idx].adj[DIR_TOP] -= ITAU * (backLattice[idx].adj[DIR_TOP] - topEq);
			backLattice[idx].adj[DIR_LEFT] -= ITAU * (backLattice[idx].adj[DIR_LEFT] - leftEq);
			backLattice[idx].adj[DIR_BOTTOM] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM] - bottomEq);
			backLattice[idx].adj[DIR_TOP_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[idx].adj[DIR_TOP_LEFT] -= ITAU * (backLattice[idx].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_LEFT] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}

		}
	}
}

void LBM2D_1D_indices::collisionStepStreamlined() {


	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			float macroDensity = calculateMacroscopicDensity(x, y);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);

			int idx = getIdx(x, y);
			velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);


			velocityArrows.push_back(glm::vec3(x, y, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[idx] * 5.0f, -1.0f) + glm::vec3(x, y, 0.0f));


			// let's find the equilibrium
			float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
			float leftTermAxis = WEIGHT_AXIS * macroDensity;
			float leftTermDiagonal = WEIGHT_DIAGONAL * macroDensity;

			// optimize these operations later

			float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
			float thirdTerm = 1.5f * macroVelocityDot;

			float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

			// this can all be rewritten into arrays + for cycles!
			float rightEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);
			float topEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);
			float leftEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);
			float bottomEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);
			float topRightEq = leftTermDiagonal * (1.0f + 3.0f * (macroVelocity.x + macroVelocity.y) +
												   4.5f * (macroVelocity.x + macroVelocity.y) * (macroVelocity.x + macroVelocity.y) - thirdTerm);
			float topLeftEq = leftTermDiagonal * (1.0f + 3.0f * (-macroVelocity.x + macroVelocity.y) +
												  4.5f * (-macroVelocity.x + macroVelocity.y) * (-macroVelocity.x + macroVelocity.y) - thirdTerm);
			float bottomLeftEq = leftTermDiagonal * (1.0f + 3.0f * (-macroVelocity.x - macroVelocity.y) +
													 4.5f * (-macroVelocity.x - macroVelocity.y) * (-macroVelocity.x - macroVelocity.y) - thirdTerm);
			float bottomRightEq = leftTermDiagonal * (1.0f + 3.0f * (macroVelocity.x - macroVelocity.y) +
													  4.5f * (macroVelocity.x - macroVelocity.y) * (macroVelocity.x - macroVelocity.y) - thirdTerm);

			backLattice[idx].adj[DIR_MIDDLE] -= ITAU * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);
			backLattice[idx].adj[DIR_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_RIGHT] - rightEq);
			backLattice[idx].adj[DIR_TOP] -= ITAU * (backLattice[idx].adj[DIR_TOP] - topEq);
			backLattice[idx].adj[DIR_LEFT] -= ITAU * (backLattice[idx].adj[DIR_LEFT] - leftEq);
			backLattice[idx].adj[DIR_BOTTOM] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM] - bottomEq);
			backLattice[idx].adj[DIR_TOP_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[idx].adj[DIR_TOP_LEFT] -= ITAU * (backLattice[idx].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_LEFT] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}

		}
	}


}



void LBM2D_1D_indices::moveParticles() {


	glm::vec2 adjVelocities[4];
	for (int i = 0; i < NUM_PARTICLES; i++) {
		float x = particleVertices[i].x;
		float y = particleVertices[i].y;


		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[getIdx(leftX, topY)];
		adjVelocities[1] = velocities[getIdx(rightX, topY)];
		adjVelocities[2] = velocities[getIdx(leftX, bottomY)];
		adjVelocities[3] = velocities[getIdx(rightX, bottomY)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);

#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		particleArrows.push_back(particleVertices[i]);
#endif
		//if (particleSystem->streamLines[i].size() >= MAX_STREAMLINE_LENGTH) {
		//	particleSystem->streamLines[i].pop_front();
		//}
		//particleSystem->streamLines[i].push_back(particleVertices[i]);
		particleSystem->streamLines[i * MAX_STREAMLINE_LENGTH + streamLineCounter] = particleVertices[i];


		particleVertices[i] += glm::vec3(finalVelocity, 0.0f);
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		glm::vec3 tmp = particleVertices[i] + 10.0f * glm::vec3(finalVelocity, 0.0f);
		particleArrows.push_back(tmp);
#endif


		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= GRID_HEIGHT - 1) {
#ifdef MIRROR_SIDES
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1) {
				particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
				if (respawnIndex >= GRID_HEIGHT - 1) {
					respawnIndex = 0;
				}
			} else {
				particleVertices[i] = glm::vec3(x, (int)(particleVertices[i].y + GRID_HEIGHT - 1) % (GRID_HEIGHT - 1), 0.0f);
			}

#else
			particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
			if (respawnIndex >= GRID_HEIGHT - 1) {
				respawnIndex = 0;
			}
#endif
			for (int k = 0; k < MAX_STREAMLINE_LENGTH; k++) {
				particleSystem->streamLines[i * MAX_STREAMLINE_LENGTH + k] = particleVertices[i];
			}

		}

	}
	streamLineCounter++;
	if (streamLineCounter > MAX_STREAMLINE_LENGTH) {
		streamLineCounter = 0;
	}
}

void LBM2D_1D_indices::updateInlets() {


	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = glm::vec3(1.0f, 0.0f, 0.0f);

	// let's find the equilibrium
	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermDiagonal = weightDiagonal * macroDensity;

	// optimize these operations later

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	// this can all be rewritten into arrays + for cycles!
	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	for (int y = 0; y < GRID_HEIGHT; y++) {
		int idx = getIdx(0, y);
		backLattice[idx].adj[DIR_MIDDLE] = middleEq;
		backLattice[idx].adj[DIR_RIGHT] = rightEq;
		backLattice[idx].adj[DIR_TOP] = topEq;
		backLattice[idx].adj[DIR_LEFT] = leftEq;
		backLattice[idx].adj[DIR_BOTTOM] = bottomEq;
		backLattice[idx].adj[DIR_TOP_RIGHT] = topRightEq;
		backLattice[idx].adj[DIR_TOP_LEFT] = topLeftEq;
		backLattice[idx].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		backLattice[idx].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
		//velocities[idx] = macroVelocity;
	}




}

void LBM2D_1D_indices::updateInlets(Node *lattice) {



	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = glm::vec3(1.0f, 0.0f, 0.0f);

	// let's find the equilibrium
	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermDiagonal = weightDiagonal * macroDensity;

	// optimize these operations later

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	// this can all be rewritten into arrays + for cycles!
	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			int idx = getIdx(x, y);
			lattice[idx].adj[DIR_MIDDLE] = middleEq;
			lattice[idx].adj[DIR_RIGHT] = rightEq;
			lattice[idx].adj[DIR_TOP] = topEq;
			lattice[idx].adj[DIR_LEFT] = leftEq;
			lattice[idx].adj[DIR_BOTTOM] = bottomEq;
			lattice[idx].adj[DIR_TOP_RIGHT] = topRightEq;
			lattice[idx].adj[DIR_TOP_LEFT] = topLeftEq;
			lattice[idx].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
			lattice[idx].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
			for (int i = 0; i < 9; i++) {
				if (lattice[idx].adj[i] < 0.0f) {
					lattice[idx].adj[i] = 0.0f;
				} else if (lattice[idx].adj[i] > 1.0f) {
					lattice[idx].adj[i] = 1.0f;
				}
			}
			//velocities[idx] = macroVelocity;
		}
	}

}

void LBM2D_1D_indices::updateColliders() {

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			int idx = getIdx(x, y);

			if (/*testCollider[row][col] ||*/ /*y == 0 || y == GRID_HEIGHT - 1 ||*/ tCol->area[idx]) {


				float right = backLattice[idx].adj[DIR_RIGHT];
				float top = backLattice[idx].adj[DIR_TOP];
				float left = backLattice[idx].adj[DIR_LEFT];
				float bottom = backLattice[idx].adj[DIR_BOTTOM];
				float topRight = backLattice[idx].adj[DIR_TOP_RIGHT];
				float topLeft = backLattice[idx].adj[DIR_TOP_LEFT];
				float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT];
				float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT];
				backLattice[idx].adj[DIR_RIGHT] = left;
				backLattice[idx].adj[DIR_TOP] = bottom;
				backLattice[idx].adj[DIR_LEFT] = right;
				backLattice[idx].adj[DIR_BOTTOM] = top;
				backLattice[idx].adj[DIR_TOP_RIGHT] = bottomLeft;
				backLattice[idx].adj[DIR_TOP_LEFT] = bottomRight;
				backLattice[idx].adj[DIR_BOTTOM_LEFT] = topRight;
				backLattice[idx].adj[DIR_BOTTOM_RIGHT] = topLeft;


				//float macroDensity = calculateMacroscopicDensity(x, y);
				//glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);
				//velocities[idx] = macroVelocity;

			}


		}
	}

}


void LBM2D_1D_indices::initBuffers() {


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	vector<glm::vec3> bData;
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			bData.push_back(glm::vec3(x, y, 0.0f));
		}
	}

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


	// Velocity arrows
	glGenVertexArrays(1, &velocityVAO);
	glBindVertexArray(velocityVAO);
	glGenBuffers(1, &velocityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);




	// Particle arrows
	glGenVertexArrays(1, &particleArrowsVAO);
	glBindVertexArray(particleArrowsVAO);
	glGenBuffers(1, &particleArrowsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);


	glBindVertexArray(0);


}

void LBM2D_1D_indices::initLattice() {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			int idx = getIdx(x, y);
			frontLattice[idx].adj[DIR_MIDDLE] = weightMiddle;
			for (int dir = 1; dir <= 4; dir++) {
				frontLattice[idx].adj[dir] = weightAxis;
			}
			for (int dir = 5; dir <= 8; dir++) {
				frontLattice[idx].adj[dir] = weightDiagonal;
			}
		}
	}


}

void LBM2D_1D_indices::initTestCollider() {
	tCol = new LatticeCollider(COLLIDER_FILENAME);

	cudaMemcpy(d_tCol, &tCol->area[0], sizeof(bool) * GRID_WIDTH * GRID_HEIGHT, cudaMemcpyHostToDevice);

}

void LBM2D_1D_indices::swapLattices() {
	// CPU
	Node *tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;

	// GPU
	tmp = d_frontLattice;
	d_frontLattice = d_backLattice;
	d_backLattice = tmp;

}

float LBM2D_1D_indices::calculateMacroscopicDensity(int x, int y) {

	float macroDensity = 0.0f;
	int idx = getIdx(x, y);
	for (int i = 0; i < 9; i++) {
		macroDensity += backLattice[idx].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM2D_1D_indices::calculateMacroscopicVelocity(int x, int y, float macroDensity) {
	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	int idx = getIdx(x, y);
	macroVelocity += vRight * backLattice[idx].adj[DIR_RIGHT];
	macroVelocity += vTop * backLattice[idx].adj[DIR_TOP];
	macroVelocity += vLeft * backLattice[idx].adj[DIR_LEFT];
	macroVelocity += vBottom * backLattice[idx].adj[DIR_BOTTOM];
	macroVelocity += vTopRight * backLattice[idx].adj[DIR_TOP_RIGHT];
	macroVelocity += vTopLeft * backLattice[idx].adj[DIR_TOP_LEFT];
	macroVelocity += vBottomLeft * backLattice[idx].adj[DIR_BOTTOM_LEFT];
	macroVelocity += vBottomRight * backLattice[idx].adj[DIR_BOTTOM_RIGHT];
	macroVelocity /= macroDensity;


	return macroVelocity;
}
