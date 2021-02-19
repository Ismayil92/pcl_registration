#define GLM_ENABLE_EXPERIMENTAL
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h> 
#include <pcl/features/normal_3d.h> 
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/console/parse.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <glm/gtx/string_cast.hpp>
#include "pcl_register_CUDA.hpp"
#include "scanmatch.h"
// CONSTANTS of ICP CUDA
#define VISUALIZE 1
#define STEP true
#define CPU false
#define GPU_NAIVE false
#define GPU_OCTREE false
#define GPU_KDTREE true
#define MODEL true
#define UNIFORM_GRID 0
#define COHERENT_GRID 0
GLFWwindow *window;
int N_FOR_VIS = 5000;
// glm vectors for ICP CUDA
std::vector<glm::vec3> glm_scene;
std::vector<glm::vec3> glm_ref;
// KDTree Global variables
glm::vec4 *YbufferTree;
glm::ivec3 *track;
int p_treesize;
int p_tracksize;
// Point Cloud pointers
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_transformed_dl(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_transformed_ICP(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

// Reference to Rough Transformation matrix coming from Pose Estimation using DL
Eigen::Matrix4f Rough_T_matrix_from_pose_est = Eigen::Matrix4f::Identity();
Eigen::Matrix4f icp_cuda_transformation; 
double leaf_size = 0.005f;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{
  pcl::fromROSMsg(*input, *scene_cloud_ptr); 
}
pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

void callbackMatrix(const geometry_msgs::Transform::ConstPtr& msg)
{ 
    Eigen::Matrix<double,3,3> tmat;
    float x = msg->translation.x;
    float y = msg->translation.y;
    float z = msg->translation.z;
    float beta_x = msg->rotation.x;
    float beta_y = msg->rotation.y;
    float beta_z = msg->rotation.z;
    float beta_w = msg->rotation.w;
    tf::Quaternion quat = tf::Quaternion(beta_x,beta_y,beta_z,beta_w);
    
    tf::Matrix3x3 rot(quat);
    //rot.getRotation(quat);
    tf::matrixTFToEigen(rot,tmat);  
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        { 
          Rough_T_matrix_from_pose_est(i,j) = tmat(i,j);
        }
    }
    Rough_T_matrix_from_pose_est(0,3) = x;
    Rough_T_matrix_from_pose_est(1,3) = y;
    Rough_T_matrix_from_pose_est(2,3) = z;
    Rough_T_matrix_from_pose_est(3,0) = 0;
    Rough_T_matrix_from_pose_est(3,1) = 0;
    Rough_T_matrix_from_pose_est(3,2) = 0;
    Rough_T_matrix_from_pose_est(3,3) = 1;      
}
std::vector<glm::vec3> pcl_to_glmvec(pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_pcl)
{
  std::vector<glm::vec3> coords;
  int i=0;
  while(i<ptr_pcl->size())
  {
	coords.push_back(glm::vec3(ptr_pcl->points[i].x, ptr_pcl->points[i].y, ptr_pcl->points[i].z));
	i++;
  } 
  return coords;
}

void runICPCUDA(std::vector<glm::vec3> glm_scene_vec, std::vector<glm::vec3> glm_ref_vec, glm::mat3 &R, glm::vec3 &t, int cycle) 
{ 
  // Initialization
  int n_points_scene = glm_scene_vec.size();  
  int n_points_ref = glm_ref_vec.size();
  #if GPU_NAIVE
    ScanMatch::initSimulationGPU(n_points_scene, n_points_ref, glm_scene_vec, glm_ref_vec);
  #endif 
  #if GPU_OCTREE 
    ScanMatch::initSimulationGPUOCTREE(n_points_scene, n_points_ref, glm_scene_vec, glm_ref_vec);  
  #endif
  #if GPU_KDTREE 
    ScanMatch::initSimulationGPUKDTREE(n_points_scene, n_points_ref, glm_scene_vec, glm_ref_vec, YbufferTree, track, p_treesize, p_tracksize);
  #endif
  updateCamera();
  glEnable(GL_DEPTH_TEST);
  std::cout<<"Now started to buffer"<<std::endl;
  // Run CUDA
  float *dptr = NULL;
  float *dptrVertPositions = NULL;
  float *dptrVertVelocities = NULL;
  cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
  cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);   
  #if GPU_NAIVE
    std::cout<<" started to run stepICPGPU_NAIVE"<<std::endl;
    ScanMatch::stepICPGPU_NAIVE(R,t);  
  #endif
  #if GPU_OCTREE
    std::cout<<" started to run stepICPGPU_OCTREE"<<std::endl;
    ScanMatch::stepICPGPU_OCTREE(R,t);
  #endif
  #if GPU_KDTREE 
    std::cout<<" started to run stepICPGPU_KDTREE"<<std::endl;
    ScanMatch::stepICPGPU_KDTREE(R,t);
  #endif
  #if VISUALIZE  
	  ScanMatch::copyPointCloudToVBO(dptrVertPositions, dptrVertVelocities, CPU);
  #endif
  // unmap buffer object
  cudaGLUnmapBufferObject(boidVBO_positions);
  cudaGLUnmapBufferObject(boidVBO_velocities);  
}


int main (int argc, char** argv)
{ 

  int cycle = 0;
  const std::string PCD_PATH = "/home/dlar/catkin_ws/body_visible_rotated.pcd";
  
  // Initialize ROS
  ros::init (argc, argv, "point_cloud_register_using_detection");
  ros::NodeHandle nh;
  ros::Rate r(10);
  
  // Create a ROS subscriber for the input point cloud from the sensor
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, cloud_cb);  
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2> ("xtion/depth_registered/registered_points", 1);
  ros::Subscriber T_matrix_sub = nh.subscribe<geometry_msgs::Transform>("/transformation_wrt_depth_optical_frame", 1, callbackMatrix);
 
  // Ros Message to publish
  sensor_msgs::PointCloud2 output;
  // Import reference point clouds data
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (PCD_PATH, *ref_cloud_ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file reference point cloud. \n");
    return (-1);
  }  
  
  // Downsampling Reference Cloud ---------------------------------------
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (ref_cloud_ptr);
  sor.setLeafSize (leaf_size, leaf_size, leaf_size);
  sor.filter (*ref_cloud_ptr);


  float depth_limit;
  float height_limit;
  float width_limit;

  //======================= CUDA SETTINGS =============================== 
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if(gpuDevice>device_count)
  {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }

  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  
  int major = deviceProp.major;
  int minor = deviceProp.minor;  
  const char *projectName = "pcl_registration_cuda";
  std::ostringstream ss;
  std::string deviceName;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();  

   // Window setup stuff  
  
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }
  
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  
  if (glewInit() != GLEW_OK) {
    //return false;
    std::cout<<"init failed"<<std::endl;
  }
  
  std::cout<<"CUDA Device is set to zero"<<std::endl; 
  cudaGLSetGLDevice(0);
  initShaders(program);

  // glm Rotation and Translation variables for ICP CUDA
  glm::mat3 R;
  glm::vec3 t;

 

  bool flag_first_iteration = true; 

  std::cout<<"ROS Cycle starts"<<std::endl; 
  while(ros::ok())
  { 
    cycle++;
    
    if(scene_cloud_ptr->size() > 0)
    { 
      
      // ============================ FILTERING SCENE POINT CLOUD ===============================
      if(flag_first_iteration == true)
      {      
        // Save unfiltered scene cloud    
        pcl::io::savePCDFileASCII("data_scene_unfiltered.pcd", *scene_cloud_ptr); 
      }
      // ...removing distant points      
      pcl::PassThrough<pcl::PointXYZ> pass;
      pass.setInputCloud (scene_cloud_ptr);
      pass.setFilterFieldName ("z");
      depth_limit = Rough_T_matrix_from_pose_est(2,3);
      pass.setFilterLimits (0, depth_limit+0.15);
      pass.filter (*scene_cloud_ptr);
      pass.setFilterFieldName("x");
      width_limit = Rough_T_matrix_from_pose_est(0,3);
      pass.setFilterLimits(width_limit-0.40,width_limit+0.40);
      pass.filter(*scene_cloud_ptr);  
      pass.setFilterFieldName("y");
      height_limit = Rough_T_matrix_from_pose_est(1,3);
      pass.setFilterLimits(height_limit-0.20, height_limit+0.75);
      pass.filter(*scene_cloud_ptr);  
      pass.setFilterFieldName("z");
      pass.setNegative (true);
      pass.setFilterLimits(0,depth_limit-0.15);
      pass.filter(*scene_cloud_ptr);
           
      // Downsampling scene point cloud
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud (scene_cloud_ptr);
      sor.setLeafSize (leaf_size, leaf_size, leaf_size);
      sor.filter (*scene_cloud_ptr);  
      // Remove Statistical Outliers -------------------------------------
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_statistic_removal;
      sor_statistic_removal.setInputCloud (scene_cloud_ptr);
      sor_statistic_removal.setMeanK (50);
      sor_statistic_removal.setStddevMulThresh(1.0);
      sor_statistic_removal.filter (*scene_cloud_ptr); 

      // Save filtered scene cloud
      pcl::io::savePCDFileASCII("data_scene_filtered.pcd", *scene_cloud_ptr); 
     
      // ========================== FIRST PROJECTION with Rough Transformation Matrix ====================================  
      if(flag_first_iteration == true)
      {
        pcl::transformPointCloud(*ref_cloud_ptr, *ref_cloud_transformed_dl, Rough_T_matrix_from_pose_est); 
        // Save the scene after initial transformation
        pcl::io::savePCDFileASCII("data_ref_transformed_dl.pcd", *ref_cloud_transformed_dl);
        std::cout << "Rought Transformation coming from Pose Estimation" << std::endl << Rough_T_matrix_from_pose_est<< std::endl;
      }
      
      // ========================= ICP ITERATION ==========================================================================
      glm_scene = pcl_to_glmvec(scene_cloud_ptr);
      // Filling source cloud vector 
      if(flag_first_iteration == true) 
      {
        // Converting filtered ref cloud to glm cloud for initial transformation
        glm_ref = pcl_to_glmvec(ref_cloud_transformed_dl);
      }
      else
      {
        // Converting filtered scene cloud to glm cloud   
        glm_ref = pcl_to_glmvec(ref_cloud_transformed_ICP);
      }
      #if GPU_KDTREE
       // Build KDTREE   
        std::cout<<"Building KDSEARCH tree for Target (REFERENCE) cloud"<< std::endl;
        int size = KDTree::nextPowerOf2(2*(glm_ref.size()+1)); // to store nulls for leaf nodes 
        std::vector<glm::vec4> glm_ref_Tree(size, glm::vec4(0.0f));    
  
        int sz = (int)log2(size); 
        int X = (int)glm_scene.size();
        std::vector<glm::ivec3> tk(X*sz, glm::ivec3(0,0,0));
        track = &tk[0];   
        KDTree::initCpuKDTree(glm_ref, glm_ref_Tree);
        YbufferTree = &glm_ref_Tree[0];
        p_treesize = size;        
        p_tracksize = X*sz;
      #endif
      
      // -----------------  Initialize and run ICP CUDA implementation ------------------
      N_FOR_VIS = glm_scene.size()+glm_ref.size();          
      initVAO(N_FOR_VIS);
      std::cout<<"initVAO ends to be processed"<<std::endl;      
      cudaGLRegisterBufferObject(boidVBO_positions);
      cudaGLRegisterBufferObject(boidVBO_velocities);
      // Run ICP CUDA algorithm      
      runICPCUDA(glm_scene, glm_ref, R, t, cycle);
      std::cout<<"now prepares transformation matrix" << std::endl;
      //Eigen matrix for ICP CUDA transformation to be filled from glm results           
      icp_cuda_transformation(0,0) = R[0][0];
      icp_cuda_transformation(0,1) = R[0][1];
      icp_cuda_transformation(0,2) = R[0][2];
      icp_cuda_transformation(1,0) = R[1][0];
      icp_cuda_transformation(1,1) = R[1][1];
      icp_cuda_transformation(1,2) = R[1][2];
      icp_cuda_transformation(2,0) = R[2][0];
      icp_cuda_transformation(2,1) = R[2][1];
      icp_cuda_transformation(2,2) = R[2][2];
      icp_cuda_transformation(0,3) = t[0];
      icp_cuda_transformation(1,3) = t[1];
      icp_cuda_transformation(2,3) = t[2];
      icp_cuda_transformation(3,0) = 0;
      icp_cuda_transformation(3,1) = 0;
      icp_cuda_transformation(3,2) = 0;
      icp_cuda_transformation(3,3) = 1; 

      if(flag_first_iteration == true) // for initial iteration
      {
        pcl::transformPointCloud(*ref_cloud_transformed_dl, *ref_cloud_transformed_ICP, icp_cuda_transformation.inverse()); 
      }
      else //
      {
        pcl::transformPointCloud(*ref_cloud_transformed_ICP, *ref_cloud_transformed_ICP, icp_cuda_transformation.inverse());
        // Save multiple iteration of ICP CUDA
        pcl::io::savePCDFileASCII("data_cuda_next_iterations.pcd", *ref_cloud_transformed_ICP);
      }     
      flag_first_iteration = false;  
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);      
      #if VISUALIZE
      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);
      glUseProgram(0);
      glBindVertexArray(0);
      glfwSwapBuffers(window);
      #endif 
      
      
      ScanMatch::endSimulation(); 
      
      //glfwDestroyWindow(window);
	  //glfwTerminate();  
              
    }  
   
  
    pcl::toROSMsg(*ref_cloud_transformed_ICP,output);
    output.header.frame_id = "camera_depth_optical_frame";
    output.header.stamp = ros::Time::now();
    pub.publish (output);        

    ros::spinOnce();
    r.sleep();
    
  }
  
  return 0;
}
