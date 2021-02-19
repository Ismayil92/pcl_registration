#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "scanmatch.h"
#include "svd3.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef clamp
#define clamp(x, lo, hi) (x < lo) ? lo : (x > hi) ? hi : x
#endif

#ifndef wrap
#define wrap(x, lo, hi) (x < lo) ? x + (hi - lo) : (x > hi) ? x - (hi - lo) : x
#endif


#define checkCUDAErrorWithLine(msg) utilityCore::checkCUDAError(msg, __LINE__)

#define DEBUG false

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 64

/*! Size of the starting area in simulation space. 
 * FOR SINE TEST: 2.f
 * FOR ELEPHANT OBJ: 
 * FOR BUDDHA OBJ: 1 << 2;
 * FOR WAYMO DATASET: 1 << 5;
*/

#define scene_scale 1 << 4

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
int numScene;
int numRef;



dim3 threadsPerBlock(blockSize);

glm::vec3 *dev_pos;
glm::vec3 *dev_rgb;

pointcloud* target_pc;
pointcloud* src_pc;

//OCTREE pointer (all octnodes lie in device memory)
Octree* octree;
OctNodeGPU *dev_octoNodes;
glm::vec3 *dev_octoCoords;

//KDTREE pointer
KDTree::Node *dev_kd;


__host__ __device__ bool sortFuncX(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.x < p2.x;
}
__host__ __device__ bool sortFuncY(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.y < p2.y;
}
__host__ __device__ bool sortFuncZ(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.z < p2.z;
}



/******************
* initSimulation *
******************/
/**
* Initialize memory, update some globals
*/
void ScanMatch::initSimulationCPU(int N, std::vector<glm::vec3> coords) {
  numObjects = N;

  //Setup and initialize source and target pointcloud
  src_pc = new pointcloud(false, numObjects, false);
  src_pc->initCPU();
  target_pc = new pointcloud(true, numObjects, false);
  target_pc->initCPU();
}

void ScanMatch::initSimulationGPU(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref) {  
  numObjects = num_scene + num_ref;
  numScene = num_scene;
  numRef = num_ref;
  //Setup and initialize source and target pointcloud
  src_pc = new pointcloud(false, numScene, true);
  src_pc->initGPU(ptr_scene);
  target_pc = new pointcloud(true, numRef, true);
  target_pc->initGPU(ptr_ref);
}

void ScanMatch::initSimulationGPUOCTREE(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref) {
  numObjects = num_scene + num_ref;
  numScene = num_scene;
  numRef = num_ref;
  
  //First create the Octree 
  octree = new Octree(glm::vec3(0.f, 0.f, 0.f), 1<<2, ptr_scene);
  octree->create();
  octree->compact();

  //Extract Final Data from Octree
  int numNodes = octree->gpuNodePool.size();
  glm::vec3* octoCoords = octree->gpuCoords.data();
  OctNodeGPU* octoNodes = octree->gpuNodePool.data();

  //Send stuff to device
  cudaMalloc((void**)&dev_octoNodes, numNodes * sizeof(OctNodeGPU));
  utilityCore::checkCUDAError("cudaMalloc octor failed", __LINE__);
  cudaMemcpy(dev_octoNodes, octoNodes, numNodes * sizeof(OctNodeGPU), cudaMemcpyHostToDevice);
  utilityCore::checkCUDAError("cudaMemcpy octoNodes failed", __LINE__);
  src_pc = new pointcloud(false, numScene, true);
  src_pc->initGPU(ptr_scene);
  target_pc = new pointcloud(true, numRef, true);
  target_pc->initGPU(ptr_ref);
}

void ScanMatch::initSimulationGPUKDTREE(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref, glm::vec4* YbufferTree, glm::ivec3* _track, int _treesize, int _tracksize) {
	numObjects = num_scene + num_ref;
	numScene = num_scene;
	numRef = num_ref;  	
	// KDTree
	
	KDTree::Node *kd = new KDTree::Node[numScene];
	KDTree::Create(ptr_scene, kd);
	// Memory allocations 
	cudaMalloc((void**)&dev_kd, numScene * sizeof(KDTree::Node));
	utilityCore::checkCUDAError("cudaMalloc dev_kd",__LINE__);
	// Copy data of CPU arrays to pointers of GPU arrays
	cudaMemcpy(dev_kd, kd, numScene * sizeof(KDTree::Node), cudaMemcpyHostToDevice);
	utilityCore::checkCUDAError("cudaMemcpy dev_kd failed",__LINE__);
	// create cloud pointers in GPU 
	src_pc = new pointcloud(false, numScene, true); //first parameter mean if it is target point cloud  
	src_pc->initGPU(ptr_scene);
	target_pc = new pointcloud(true, numRef, true);
	target_pc->initGPU(ptr_ref);	
}




/******************
* copyPointCloudToVBO *
******************/

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void ScanMatch::copyPointCloudToVBO(float *vbodptr_positions, float *vbodptr_rgb, bool usecpu) {

	if (usecpu) { //IF CPU
	  src_pc->pointCloudToVBOCPU(vbodptr_positions, vbodptr_rgb, scene_scale);
	  target_pc->pointCloudToVBOCPU(vbodptr_positions + 4*numRef, vbodptr_rgb + 4*numRef, scene_scale);
	}
	else { //IF GPU
		src_pc->pointCloudToVBOGPU(numScene,vbodptr_positions, vbodptr_rgb, scene_scale);
		target_pc->pointCloudToVBOGPU(numScene,vbodptr_positions + 4*numScene, vbodptr_rgb + 4*numScene, scene_scale);
	}
}


/******************
* stepSimulation *
******************/

void ScanMatch::endSimulation() {
	src_pc->~pointcloud();
	target_pc->~pointcloud();
	cudaFree(dev_kd);
	
}

/******************
* CPU SCANMATCHING *
******************/

/**
 * Main Algorithm for Running ICP on the CPU
 * Finds homogenous transform between src_pc and target_pc 
*/
void ScanMatch::stepICPCPU() {
	//1: Find Nearest Neigbors and Reshuffle
	float* dist = new float[numObjects];
	int* indicies = new int[numObjects];
#if DEBUG
	printf("NEAREST NEIGHBORS \n");
#endif // DEBUG

	auto start = std::chrono::high_resolution_clock::now();
	ScanMatch::findNNCPU(src_pc, target_pc, dist, indicies, numObjects);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
	std::cout << duration.count() << std::endl;

#if DEBUG
	printf("RESHUFFLE\n");
#endif // DEBUG

	ScanMatch::reshuffleCPU(target_pc, indicies, numObjects);

	//2: Find Best Fit Transformation
	glm::mat3 R;
	glm::vec3 t;
	ScanMatch::bestFitTransform(src_pc, target_pc, numObjects, R, t);


	//3: Update each src_point
	glm::vec3* src_dev_pos = src_pc->dev_pos;
	for (int i = 0; i < numObjects; ++i) {
		src_dev_pos[i] = glm::transpose(R) * src_dev_pos[i] + t;
	}
}

/**
 * Finds Nearest Neighbors of target pc in src pc
 * @args: src, target -> PointClouds w/ filled dev_pos
 * @returns: 
	* dist -> N array -> ith index = dist(src[i], closest_point in target)
	* indicies -> N array w/ ith index = index of the closest point in target to src[i]
*/
void ScanMatch::findNNCPU(pointcloud* src, pointcloud* target, float* dist, int *indicies, int N) {
	glm::vec3* src_dev_pos = src->dev_pos;
	glm::vec3* target_dev_pos = target->dev_pos;
	for (int src_idx = 0; src_idx < N; ++src_idx) { //Iterate through each source point
		glm::vec3 src_pt = src_dev_pos[src_idx];
		float minDist = INFINITY;
		int idx_minDist = -1;
		for (int tgt_idx = 0; tgt_idx < N; ++tgt_idx) { //Iterate through each tgt point and find closest
			glm::vec3 tgt_pt = target_dev_pos[tgt_idx];
			float d = glm::distance(src_pt, tgt_pt);
			if (d < minDist) {
				minDist = d;
				idx_minDist = tgt_idx;
			}
		}
		//Update dist and indicies

#if DEBUG
		printf("IDX: %d - MINDIST %f\n", src_idx, minDist);
		printf("IDX: %d - indicies %d\n", src_idx, idx_minDist);
#endif // DEBUG

		dist[src_idx] = minDist;
		indicies[src_idx] = idx_minDist;
	}
}

/**
 * Reshuffles pointcloud a as per indicies, puts these in dev_matches
 * NOT ONE TO ONE SO NEED TO MAKE A COPY!
*/
void ScanMatch::reshuffleCPU(pointcloud* a, int* indicies, int N) {
	glm::vec3 *a_dev_matches = a->dev_matches;
	glm::vec3 *a_dev_pos = a->dev_pos;
	for (int i = 0; i < N; ++i) {
		a_dev_matches[i] = a_dev_pos[indicies[i]];

#if DEBUG
		printf("DEV MATCHES\n");
		utilityCore::printVec3(a->dev_matches[i]);
		printf("DEV POS\n");
		utilityCore::printVec3(a_dev_pos[i]);
#endif // DEBUG
	}
}

/**
 * Calculates transform T that maps from src to target
 * Assumes dev_matches is filled for target
*/
void ScanMatch::bestFitTransform(pointcloud* src, pointcloud* target, int N, glm::mat3 &R, glm::vec3 &t){
	glm::vec3* src_norm = new glm::vec3[N];
	glm::vec3* target_norm = new glm::vec3[N];
	glm::vec3 src_centroid(0.f);
	glm::vec3 target_centroid(0.f);
	glm::vec3* src_pos = src->dev_pos;
	glm::vec3* target_matches = target->dev_matches;

	//1:Calculate centroids and norm src and target
	for (int i = 0; i < N; ++i) {
		src_centroid += src_pos[i];
		target_centroid += target_matches[i];
	}
	src_centroid = src_centroid / glm::vec3(N);
	target_centroid = target_centroid / glm::vec3(N);

#if DEBUG
	printf("SRC CENTROID\n");
	utilityCore::printVec3(src_centroid);
	printf("TARGET CENTROID\n");
	utilityCore::printVec3(target_centroid);
#endif // DEBUG

	for (int j = 0; j < N; ++j) {
		src_norm[j] = src_pos[j]  - src_centroid;
		target_norm[j] = target_matches[j] - target_centroid;
#if DEBUG
		printf("SRC NORM IDX %d\n", j);
		utilityCore::printVec3(src_norm[j]);
		printf("TARGET NORM IDX %d\n", j);
		utilityCore::printVec3(target_norm[j]);
#endif // DEBUG
	}

	//1:Multiply src.T (3 x N) by target (N x 3) = H (3 x 3)
	float H[3][3] = { 0 };
	for (int i = 0; i < N; ++i) { //3 x N by N x 3 matmul
		for (int out_row = 0; out_row < 3; out_row++) {
			for (int out_col = 0; out_col < 3; out_col++) {
				H[out_row][out_col] += src_norm[i][out_row] * target_norm[i][out_col];
			}
		}
	}
	
#if DEBUG
	printf("H MATRIX ======================================================\n");
    std::cout << H[0][0] << " " << H[1][0] << " " << H[2][0] << " " << std::endl;
    std::cout << H[0][1] << " " << H[1][1] << " " << H[2][1] << " " << std::endl;
    std::cout << H[0][2] << " " << H[1][2] << " " << H[2][2] << " " << std::endl;
	printf("======================================================\n");
#endif // DEBUG

	//2:calculate SVD of H to get U, S & V
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	svd(H[0][0], H[0][1], H[0][2], H[1][0], H[1][1], H[1][2], H[2][0], H[2][1], H[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
		);
	glm::mat3 matU(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 matV(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

#if DEBUG
	printf("U MATRIX\n");
	utilityCore::printMat3(matU);
	printf("V MATRIX\n");
	utilityCore::printMat3(matV);
#endif // DEBUG

	//2:Rotation Matrix and Translation Vector
	R = (matU * matV);
	t = target_centroid - R * (src_centroid);

#if DEBUG
	printf("ROTATION\n");
	utilityCore::printMat3(R);
	printf("TRANSLATION\n");
	utilityCore::printVec3(t);
#endif // DEBUG
}

/******************
* GPU NAIVE SCANMATCHING *
******************/

__global__ void kernUpdatePositions(glm::vec3* src_pos, glm::mat3 R, glm::vec3 t, int N) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < N) {
	  src_pos[idx] = (R) * src_pos[idx] + t;
  }
}

/**
 * Main Algorithm for Running ICP on the GPU
 * Finds homogenous transform between src_pc and target_pc 
*/
void ScanMatch::stepICPGPU_NAIVE(glm::mat3 &R, glm::vec3 &t) {

	//cudaMalloc dist and indicies
	float* dist;
	int* indicies;

	cudaMalloc((void**)&dist, numScene * sizeof(float));
	utilityCore::checkCUDAError("cudaMalloc dist failed", __LINE__);

	cudaMalloc((void**)&indicies, numScene * sizeof(int));
	utilityCore::checkCUDAError("cudaMalloc indicies failed", __LINE__);
	cudaMemset(dist, 0, numScene * sizeof(float));
	cudaMemset(indicies, -1, numScene * sizeof(int));
	
	auto start = std::chrono::high_resolution_clock::now();
	
	//1: Find Nearest Neigbors and Reshuffle
	ScanMatch::findNNGPU_NAIVE(src_pc, target_pc, dist, indicies, numObjects, numScene, numRef);
	cudaDeviceSynchronize();
	ScanMatch::reshuffleGPU(target_pc, indicies, numScene);
	cudaDeviceSynchronize();
	//Correspondence filtering 

	//2: Find Best Fit Transformation	
	ScanMatch::bestFitTransformGPU(src_pc, target_pc, numObjects, numScene, numRef, R, t);
	cudaDeviceSynchronize();
	//3: Update each src_point via Kernel Call
	dim3 fullBlocksPerGrid((numScene + blockSize - 1) / blockSize);
	kernUpdatePositions<<<fullBlocksPerGrid, blockSize>>>(src_pc->dev_pos, R, t, numScene);	
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
	std::cout << "duration: "<< duration.count() << std::endl;
	
	//cudaFree dist and indicies
	cudaFree(dist);
	cudaFree(indicies);
}

/**
 * Main Algorithm for Running ICP on the GPU w/Octree
 * Finds homogenous transform between src_pc and target_pc 
*/
void ScanMatch::stepICPGPU_OCTREE(glm::mat3 &R, glm::vec3 &t) {
	//cudaMalloc dist and indicies
	float* dist;
	int* indicies;

	cudaMalloc((void**)&dist, numScene * sizeof(float));
	utilityCore::checkCUDAError("cudaMalloc dist failed", __LINE__);

	cudaMalloc((void**)&indicies, numScene * sizeof(int));
	utilityCore::checkCUDAError("cudaMalloc indicies failed", __LINE__);
	cudaMemset(dist, 0, numScene * sizeof(float));
	cudaMemset(indicies, -1, numScene * sizeof(int));
	
	auto start = std::chrono::high_resolution_clock::now();	
	
	//1: Find Nearest Neigbors and Reshuffle
	ScanMatch::findNNGPU_OCTREE(src_pc, target_pc, dist, indicies, numObjects, numScene, numRef, dev_octoNodes);
	cudaDeviceSynchronize();
	std::cout<<"findNN done"<<std::endl;
	// Reshuffling fills dev_matches list for target point cloud
	ScanMatch::reshuffleGPU(target_pc, indicies, numScene);
	cudaDeviceSynchronize();
	std::cout<<"reshuffle done"<<std::endl;
	//2: Find Best Fit Transformation
	ScanMatch::bestFitTransformGPU(src_pc, target_pc, numObjects, numScene, numRef, R, t);
	cudaDeviceSynchronize();
	std::cout<<"bestfit done"<<std::endl;
	
	//3: Update each src_point via Kernel Call
	//dim3 fullBlocksPerGrid((numScene + blockSize - 1) / blockSize);
	//kernUpdatePositions<<<fullBlocksPerGrid, blockSize>>>(src_pc->dev_pos, R, t, numScene);
	

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
	std::cout << "duration: " << duration.count() << std::endl;
	//cudaFree dist and indicies
	cudaFree(dist);
	cudaFree(indicies);
}

void ScanMatch::stepICPGPU_KDTREE(glm::mat3 &R, glm::vec3 &t)
{
	// CUDA Malloc dist and indices
	float* dist;
	int* indicies;

	cudaMalloc((void**)&dist, numScene * sizeof(float));
	utilityCore::checkCUDAError("cudaMalloc dist failed", __LINE__);
	cudaMalloc((void**)&indicies, numScene * sizeof(int));
	utilityCore::checkCUDAError("cudaMalloc indicies failed", __LINE__);
	// Initialization numScene * datatype  part of memory area assigned to dist and indicies pointers some values
	cudaMemset(dist, 0, numScene * sizeof(float));
	cudaMemset(indicies, -1, numScene * sizeof(int));

	auto start = std::chrono::high_resolution_clock::now();		

	// 1) Find nearest neighbours for each given source point  
	ScanMatch::findNNGPU_KDTREE(src_pc, target_pc, dist, indicies, numObjects, numScene, numRef, dev_kd);
	cudaDeviceSynchronize();
	// Inserting matched points in target cloud into new array pointer 
	ScanMatch::reshuffleGPU(target_pc, indicies, numScene);
	cudaDeviceSynchronize();
	// 2) Find best fit transformation matrix 
	ScanMatch::bestFitTransformGPU(src_pc, target_pc, numObjects, numScene, numRef, R, t);
	cudaDeviceSynchronize();
	

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); 
	std::cout << "duration: "<< duration.count() << std::endl;

	// 3) Update source point cloud 
	dim3 fullBlocksPerGrid((numScene + blockSize - 1) / blockSize);
	kernUpdatePositions<<<fullBlocksPerGrid, blockSize>>>(src_pc->dev_pos, R, t, numScene);
	

	cudaFree(dist);
	cudaFree(indicies);	
}


/**
 GPU NAIVE NN
 * Finds Nearest Neighbors of target pc in src pc
 * @args: src, target -> PointClouds w/ filled dev_pos IN GPU
 * @returns: 
	* dist -> N array -> ith index = dist(src[i], closest_point in target) (on GPU)
	* indicies -> N array w/ ith index = index of the closest point in target to src[i] (on GPU)
 * Parallely compute NN for each point in the pointcloud
 */

 __global__ void kernGPU_NAIVE(glm::vec3* src_pos, glm::vec3* target_pos, float* dist, int* indicies, int N, int num_scene, int num_ref) {	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		
	
	if (idx < num_scene)	
	{	
		
		float minDist = INFINITY;
	    float idx_minDist = -1;		
		glm::vec3 src_pt = src_pos[idx];
		for (int tgt_idx = 0; tgt_idx < num_ref; ++tgt_idx) { //Iterate through each tgt & find closest
			glm::vec3 tgt_pt = target_pos[tgt_idx];
			float d = glm::distance(src_pt, tgt_pt);			
			if (d < minDist && d < 0.5) {
				minDist = d;
				idx_minDist = tgt_idx;
				
			}
		}
		dist[idx] = minDist;
		indicies[idx] = idx_minDist;
				
	}
			
}
void ScanMatch::findNNGPU_NAIVE(pointcloud* src, pointcloud* target, float* dist, int *indicies, int N, int num_scene, int num_ref) {		
	//Launch a kernel (paralellely compute NN for each point)	
	dim3 fullBlocksPerGrid((num_scene + blockSize - 1) / blockSize);
	kernGPU_NAIVE<<<fullBlocksPerGrid,blockSize>>>(src->dev_pos, target->dev_pos, dist, indicies, N, num_scene, num_ref);	
}

/**
 GPU OCTREE NN
 * Finds Nearest Neighbors of target pc in src pc
 * @args: src, target -> PointClouds w/ filled dev_pos IN GPU
 * @returns: 
	* dist -> N array -> ith index = dist(src[i], closest_point in target) (on GPU)
	* indicies -> N array w/ ith index = index of the closest point in target to src[i] (on GPU)
 * Parallely compute NN for each point in the pointcloud
 */

 __device__ OctNodeGPU findLeafOctant(glm::vec3 src_pos, OctNodeGPU* octoNodes) {
	octKey currKey = 0;
	OctNodeGPU currNode = octoNodes[currKey];
	OctNodeGPU parentNode = currNode;
	//printf("SRC: %f, %f, %f \n", src_pos.x, src_pos.y, src_pos.z);
	while (!currNode.isLeaf) {
		//Determine which octant the point lies in (0 is bottom-back-left)
		glm::vec3 center = currNode.center;
		uint8_t x = src_pos.x > center.x;
		uint8_t y = src_pos.y > center.y;
		uint8_t z = src_pos.z > center.z;

		//printf("currKey: %d\n", currKey);
		//printf("currNodeBaseKey: %d\n", currNode.firstChildIdx);

		//Update the code
		currKey = currNode.firstChildIdx + (x + 2 * y + 4 * z);
		parentNode = currNode;
		currNode = octoNodes[currKey];
	}
	//printf("currKey: %d\n", currKey);
	//printf("currNodeBaseKey: %d\n", currNode.firstChildIdx);

	//printf("OCTANT CENTER: %f, %f, %f \n", currNode.center.x, currNode.center.y, currNode.center.z);
	//printf("Data START: %d \n", currNode.data_startIdx);
	return currNode;
}

 __global__ void kernNNGPU_OCTREE(glm::vec3* src_pos, glm::vec3* target_pos, float* dist, int* indicies, int N, int num_scene, int num_ref, OctNodeGPU* octoNodes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_scene) {
		float minDist = INFINITY;
		float idx_minDist = -1;
		glm::vec3 src_pt = src_pos[idx];
		
		//Find our leaf node and extract tgt_start and tgt_end from it
		OctNodeGPU currLeafOctant = findLeafOctant(src_pt, octoNodes);
		int tgt_start = currLeafOctant.data_startIdx;
		int tgt_end = currLeafOctant.data_startIdx + currLeafOctant.count;
  
		for (int tgt_idx = tgt_start; tgt_idx < tgt_end; ++tgt_idx) { //Iterate through each tgt & find closest
			glm::vec3 tgt_pt = target_pos[tgt_idx];
			float d = glm::distance(src_pt, tgt_pt);		  
			if (d < minDist && d < 0.6) {
				minDist = d;
				idx_minDist = tgt_idx;
			}
		}
		dist[idx] = minDist;
		indicies[idx] = idx_minDist;
	}
  }
  
void ScanMatch::findNNGPU_OCTREE(pointcloud* src, pointcloud* target, float* dist, int *indicies, int N, int num_scene, int num_ref,OctNodeGPU* octoNodes) {
	//Launch a kernel (paralellely compute NN for each point)
	dim3 fullBlocksPerGrid((num_scene + blockSize - 1) / blockSize);
	kernNNGPU_OCTREE<<<fullBlocksPerGrid, blockSize>>>(src->dev_pos, target->dev_pos, dist, indicies, N, num_scene, num_ref, octoNodes);
}

/**
  GPU KDTREE NN
 * Finds Nearest Neighbors of target pc in src pc
 * @args: src, target -> PointClouds w/ filled dev_pos IN GPU
 * @returns: 
	* dist -> N array -> ith index = dist(src[i], closest_point in target) (on GPU)
	* indicies -> N array w/ ith index = index of the closest point in target to src[i] (on GPU)
*/
__device__ float getHyperplaneDist(const glm::vec3 *pt1, const glm::vec3 *pt2, int axis, bool *branch)
{
	if (axis == 0) {
		*branch = sortFuncX(*pt1, *pt2);
		return abs(pt1->x - pt2->x);
	}
	if (axis == 1) {
		*branch = sortFuncY(*pt1, *pt2);
		return abs(pt1->y - pt2->y);
	}
	if (axis == 2) {
		*branch = sortFuncZ(*pt1, *pt2);
		return abs(pt1->z - pt2->z);
	}
	return 0;
}

__global__ void kernNNGPU_KDTREE(glm::vec3* src_pos, glm::vec3* target_pos, float* dist, int* indicies, int N, int num_scene, int num_ref, const KDTree::Node* tree)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(idx < num_scene){
		float minDist = INFINITY; 
		int idx_minDist = -1;
		glm::vec3 src_pt = src_pos[idx];
		int head = 0; 
		bool done = false;
		bool branch = false; 
		bool nodeFullyExplored = false;
		// KDTREE NN search in target cloud
		while(!done)
		{
			while(head >= 0)
			{
				const KDTree::Node test = tree[head];
				float d = glm::distance(src_pt, glm::vec3(test.value));
				if(minDist > d)
				{
					minDist = d; 
					idx_minDist = head;
					nodeFullyExplored = false;
				}
				//find branch path 
				getHyperplaneDist(&src_pt, &test.value, test.axis, &branch); 
				head = branch ? test.left : test.right;
			}
			if(nodeFullyExplored)
			{
				done = true;
			}
			else
			{
				// check if parent of best node could have better values on other branch
				const KDTree::Node parent = tree[tree[idx_minDist].parent];
				if(getHyperplaneDist(&src_pt, &glm::vec3(parent.value), parent.axis, &branch) < minDist)
				{
					head = !branch ? parent.left : parent.right;
					nodeFullyExplored = true;
				}
				else
				{
					done = true;
				}
			}
		}		
		dist[idx] = minDist;
		indicies[idx] = idx_minDist;
	}
}

void ScanMatch::findNNGPU_KDTREE(pointcloud* src, pointcloud* target, float* dist, int* indicies, int N, int num_scene, int num_ref, const KDTree::Node* tree)
{
	// Launch a kernel (parallelly compute NN for each source point
	dim3 fullBlocksPerGrid((num_scene + blockSize - 1) / blockSize);
	kernNNGPU_KDTREE<<<fullBlocksPerGrid, blockSize>>>(src->dev_pos, target->dev_pos, dist, indicies, N, num_scene, num_ref, const KDTree::Node* tree);
}

/*
 * Parallely reshuffle pos by indicies and fill matches
 */
__global__ void kernReshuffleGPU(glm::vec3* pos, glm::vec3* matches, int *indicies, int N) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < N) {
	  matches[idx] = pos[indicies[idx]];
  }
}

/**
 * Reshuffles pointcloud a as per indicies, puts these in dev_matches
 * NOT ONE TO ONE SO NEED TO MAKE A COPY!
*/
void ScanMatch::reshuffleGPU(pointcloud* a, int* indicies, int N) {
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	kernReshuffleGPU<<<fullBlocksPerGrid, blockSize>>>(a->dev_pos, a->dev_matches, indicies, N);
}

__global__ void kernComputeNormsScene(glm::vec3* src_norm, glm::vec3* pos, glm::vec3 pos_centroid, int N) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < N) {
	  src_norm[idx] = pos[idx] - pos_centroid;	  
  }
}
__global__ void kernComputeNormsRef(glm::vec3* target_norm, glm::vec3* matches, glm::vec3 matches_centroid, int N) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < N) {
		target_norm[idx] = matches[idx] - matches_centroid;
		//printf("target %f %f %f \n", target_norm[idx][0], target_norm[idx][1], target_norm[idx][2]);
		//printf("centroid %f %f %f \n", matches_centroid[0], matches_centroid[1], matches_centroid[2]);

	}
  }

__global__ void kernComputeHarray(glm::mat3* Harray, glm::vec3* src_norm, glm::vec3* target_norm, int N) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < N) {
	  Harray[idx] = glm::outerProduct(target_norm[idx],src_norm[idx]);
 }
}

/**
 * Calculates transform T that maps from src to target
 * Assumes dev_matches is filled for target
*/
void ScanMatch::bestFitTransformGPU(pointcloud* src, pointcloud* target, int N, int num_scene, int num_ref, glm::mat3 &R, glm::vec3 &t){

	glm::vec3* src_norm;
	glm::vec3* target_norm;
	glm::mat3* Harray;

	//cudaMalloc Norms and Harray
	cudaMalloc((void**)&src_norm, num_scene * sizeof(glm::vec3));
	cudaMalloc((void**)&target_norm, num_scene * sizeof(glm::vec3));
	cudaMalloc((void**)&Harray, num_scene * sizeof(glm::mat3));
	cudaMemset(Harray, 0, num_scene * sizeof(glm::mat3));


	//Thrust device pointers for calculating centroids
	thrust::device_ptr<glm::vec3> src_thrustpos(src->dev_pos);
	thrust::device_ptr<glm::vec3> target_thrustmatches(target->dev_matches);
	thrust::device_ptr<glm::mat3> harray_thrust = thrust::device_pointer_cast(Harray);
	
	//1: Calculate centroids	
	glm::vec3  src_centroid = glm::vec3(thrust::reduce(src_thrustpos, src_thrustpos + num_scene, glm::vec3(0.f), thrust::plus<glm::vec3>()));	
	cudaDeviceSynchronize();
	std::cout<<"src_centroid done"<<std::endl;
	glm::vec3 target_centroid = glm::vec3(thrust::reduce(target_thrustmatches, target_thrustmatches + num_scene, glm::vec3(0.f), thrust::plus<glm::vec3>()));	
	cudaDeviceSynchronize();
	std::cout<<"target_centroid done"<<std::endl;
	src_centroid /= glm::vec3(1.0f * num_scene);
	target_centroid /= glm::vec3(1.0f * num_scene);
	
#if DEBUG
	printf("SRC CENTROID\n");
	utilityCore::printVec3(num_ref);
	printf("TARGET CENTROID\n");
	utilityCore::printVec3(target_centroid);
#endif // DEBUG
    
	//2: Compute Norm via Kernel Call
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGridScene((num_scene + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGridRef((num_ref + blockSize - 1) / blockSize);
	kernComputeNormsScene<<<fullBlocksPerGridScene, blockSize>>>(src_norm, src->dev_pos, src_centroid, num_scene);
	std::cout<<"src normalized"<<std::endl;
	kernComputeNormsRef<<<fullBlocksPerGridScene, blockSize>>>(target_norm, target->dev_matches, target_centroid, num_scene);
	std::cout<<"target normalized"<<std::endl;
	cudaDeviceSynchronize();
	utilityCore::checkCUDAError("Compute Norms Failed", __LINE__);
	
	//3:Multiply src.T (3 x N) by target (N x 3) = H (3 x 3) via a kernel call
	kernComputeHarray<<<fullBlocksPerGridScene, blockSize>>>(Harray, src_norm, target_norm, num_scene);
	cudaDeviceSynchronize();
	std::cout<<"kernComputeArray done"<<std::endl;
	utilityCore::checkCUDAError("Compute HARRAY Failed", __LINE__);
	glm::mat3 H = thrust::reduce(harray_thrust, harray_thrust + num_scene, glm::mat3(0.f), thrust::plus<glm::mat3>());
	//glm::mat3 H = thrust::reduce(thrust::device, Harray, Harray + num_scene, glm::mat3(0.f));
	cudaDeviceSynchronize();
	std::cout<<"H done"<<std::endl;
	
	
	//4:Calculate SVD of H to get U, S & V		
	glm::mat3 U(0.0f);
	glm::mat3 S(0.0f);
	glm::mat3 V(0.0f);
	svd(H[0][0], H[0][1], H[0][2], H[1][0], H[1][1], H[1][2], H[2][0], H[2][1], H[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
		);
	
	R = glm::transpose(U) * V;
	t = target_centroid - (R * src_centroid);

	//cudaMalloc Norms and Harray
	cudaFree(src_norm);
	cudaFree(target_norm); 
	cudaFree(Harray);
}
