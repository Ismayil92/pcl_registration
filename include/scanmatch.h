#pragma once

#include <stdio.h>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <vector>
#include "pointcloud.h"
#include "utilityCore.hpp"
#include "octree.h"
#include "kdtree.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h> 

namespace ScanMatch {
    void copyPointCloudToVBO(float *vbodptr_positions, float *vbodptr_rgb, bool usecpu);
    void endSimulation();
    void unitTest();

	//CPU SCANMATCHING
    void initSimulationCPU(int N, std::vector<glm::vec3> coords);
	void stepICPCPU();
	void findNNCPU(pointcloud* src, pointcloud* target, float* dist, int* indicies, int N);
	void reshuffleCPU(pointcloud* a, int* indicies, int N);
	void bestFitTransform(pointcloud* src, pointcloud* target, int N, glm::mat3 &R, glm::vec3 &t);

	//GPU_NAIVE SCANMATCHING
    void initSimulationGPU(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref);
	void stepICPGPU_NAIVE(glm::mat3 &R, glm::vec3 &t);
	void findNNGPU_NAIVE(pointcloud* src, pointcloud* target, float* dist, int* indicies, int N, int num_scene, int num_ref);
	void reshuffleGPU(pointcloud* a, int* indicies, int N);
	void bestFitTransformGPU(pointcloud* src, pointcloud* target, int N, int num_scene, int num_ref, glm::mat3 &R, glm::vec3 &t);

	//GPU OCTREE SCANMATCHING
    void initSimulationGPUOCTREE(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref);
	void stepICPGPU_OCTREE(glm::mat3 &R, glm::vec3 &t);
	void findNNGPU_OCTREE(pointcloud* src, pointcloud* target, float* dist, int* indicies, int N,int num_scene, int num_ref, OctNodeGPU* octoNodes);

	//GPU KDTREE SCANMATCHING
	void initSimulationGPUKDTREE(int num_scene, int num_ref, std::vector<glm::vec3> ptr_scene, std::vector<glm::vec3> ptr_ref, glm::vec4* YbufferTree, glm::ivec3* _track, int _treesize, int _tracksize);
	void stepICPGPU_KDTREE(glm::mat3 &R, glm::vec3 &t);
	void findNNGPU_KDTREE(pointcloud * src, pointcloud* target, float* dist, int* indices, int N, int num_scene, int num_ref, glm::vec4* targetKDTree, glm::ivec3* targetTrack, int _treesize, int _tracksize);

} 

