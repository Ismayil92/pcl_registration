/**
* @file      kdtree.hpp
* @brief     intializer for 3-axis kd-tree
* @authors   Michael Willett
* @date      2016
* @copyright Michael Willett
*/
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <glm/vec4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace KDTree
{
	class Node {
	public:
		Node();
		Node(glm::vec3  p, int state, int source);
		
		int axis;
		int left;
		int right;
		int parent;
		glm::vec4 value;

	};

	void Create(std::vector<glm::vec3> input, Node *list);
	void InsertList(std::vector<glm::vec3> input, Node *list, int idx, int parent);
};