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
#define GPU_OCTREE true
#define GPU_KDTREE false
#define SIFT false
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
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_main_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr final_output(new pcl::PointCloud<pcl::PointXYZ>);
// Reference to Rough Transformation matrix coming from Pose Estimation using DL
Eigen::Matrix4f Rough_T_matrix_from_pose_est = Eigen::Matrix4f::Identity();
Eigen::Matrix4f icp_cuda_transformation; 
Eigen::Matrix4f gen_transformation; 
Eigen::Matrix4f T_matrix = Eigen::Matrix4f::Identity(); 
Eigen::Matrix4f shift_matrix;
// Parameters for SIFT Computation 
const float min_scale =  0.005f;
const int n_octaves = 8;
const int n_scales_per_octave = 4;
const float min_contrast =  0.001f;
double leaf_size = 0.005f;
unsigned int text_id = 0;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i)
    {
      sprintf (str, "text#%03d", i);
      viewer->removeShape (str);
    }
    text_id = 0;
  }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

    char str[512];
    sprintf (str, "text#%03d", text_id ++);
    viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}


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

pcl::PointCloud<pcl::PointNormal>::Ptr estimateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_ptr,double p_normal_radius)
{ 
  double normal_radius = p_normal_radius;
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
  std::cout<<"Started to compute normals in the function"<<std::endl;
  ne.setInputCloud(input_cloud_ptr);
  ne.setSearchMethod(tree_n);
  ne.setRadiusSearch(normal_radius);
  ne.compute(*cloud_normals);
  std::cout<<"Normals were computed"<<std::endl;
  for(int i = 0; i<cloud_normals->points.size(); ++i)
  {
    cloud_normals->points[i].x = input_cloud_ptr->points[i].x;
    cloud_normals->points[i].y = input_cloud_ptr->points[i].y;
    cloud_normals->points[i].z = input_cloud_ptr->points[i].z;
  }
  std::cout<<"Cloud normals are "<<cloud_normals->points.size()<<std::endl;
  return cloud_normals;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr estimateKeypointsSIFT(pcl::PointCloud<pcl::PointNormal>::Ptr input_cloud_normal_ptr)
{
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());

  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(input_cloud_normal_ptr);
  sift.compute(result);

  // Copying the pointwithscale to pointxyz so as visualize the cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *cloud_temp_ptr);
  std::cout << "SIFT keypoints in the cloud are " << cloud_temp_ptr->points.size () << std::endl;
  return cloud_temp_ptr;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhExtractor(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_filtered_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_ptr, pcl::PointCloud<pcl::PointNormal>::Ptr input_normals_ptr, double p_fpfh_radius)
{
  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
  for (int i = 0; i < input_normals_ptr->points.size(); i++)
  {
    if (!pcl::isFinite<pcl::PointNormal>(input_normals_ptr->points[i]))
    {
      PCL_WARN("normals[%d] is not finite\n", i);
    }
  } 
  double fpfh_radius = p_fpfh_radius;
  fpfh.setInputCloud(input_cloud_ptr);
  fpfh.setSearchSurface(input_cloud_filtered_ptr);
  fpfh.setInputNormals(input_normals_ptr);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
  fpfh.setSearchMethod(tree);
  // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
  // Radius here should be bigger than that radius used in estimation of normals
  fpfh.setRadiusSearch(fpfh_radius);
  fpfh.compute(*fpfhs);
  std::cout<<"FPFHs points size = "<<fpfhs->points.size()<<std::endl;
  std::cout<<"Input keypoint clouds size = "<<input_cloud_ptr->points.size()<<std::endl;
  return fpfhs;
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
  const std::string PCD_MAIN_PATH = "/home/dlar/catkin_ws/body_visible_rotated.pcd";
  
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
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (PCD_MAIN_PATH, *ref_main_cloud_ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file main reference point cloud \n");
    return (-1);
  }  

  // Downsampling Reference Cloud ---------------------------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (ref_cloud_ptr);
  sor.setLeafSize (leaf_size, leaf_size, leaf_size);
  sor.filter (*ref_cloud_filtered_ptr);

   
  
  pcl::PointCloud<pcl::PointNormal>::Ptr ref_cloud_normal_ptr(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr scene_cloud_normal_ptr(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_SIFT_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_SIFT_keypoints_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr ref_fpfh_ptr(new pcl::PointCloud<pcl::FPFHSignature33>());
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_fpfh_ptr(new pcl::PointCloud<pcl::FPFHSignature33>());

  std::cout<<"Started to compute normals and keypoints of the reference object."<<std::endl;

  // Setting normal and fpfh search radius
  double normal_radius_ = 0.02f;
  double fpfh_radius_ = 0.02f;
  //Estimating features and normals of Reference Cloud
  ref_cloud_normal_ptr = estimateNormals(ref_cloud_filtered_ptr,normal_radius_);
  ref_SIFT_keypoints_ptr = estimateKeypointsSIFT(ref_cloud_normal_ptr);
  ref_fpfh_ptr = fpfhExtractor(ref_cloud_filtered_ptr,ref_SIFT_keypoints_ptr,ref_cloud_normal_ptr,fpfh_radius_);
  float depth_limit = 1.2;
  float height_limit = 1.5;
  float width_limit = 0.40;

  // Point cloud pointers for scene clouds and result of icp. 
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed_dl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed_cuda(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_ref_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // CUDA settings  
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
  
  // Convertng reference cloud to glm vector
  glm_ref = pcl_to_glmvec(ref_cloud_filtered_ptr);

  // Build KDTREE 
  #if GPU_KDTREE
    std::cout<<"Building KDSEARCH tree for Target (REFERENCE) cloud"<< std::endl;
    int size = KDTree::nextPowerOf2(2*(glm_ref.size()+1)); // to store nulls for leaf nodes 
    std::vector<glm::vec4> glm_ref_Tree(size, glm::vec4(0.0f));
    
  #endif
    

  std::cout<<"ROS Cycle starts"<<std::endl; 
  while(ros::ok())
  { 
    cycle++;
    
    if(cycle>1)
    { 
      if(cycle<3)
      {      
        // Save unfiltered scene cloud    
        pcl::io::savePCDFileASCII("data_scene_unfiltered.pcd", *scene_cloud_ptr); 
      }
      // First filter scene cloud data ---------------------------------
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
      pass.setFilterLimits(0,depth_limit-0.05);
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

      // if sift feature detector activated and used instead of deep learning matrix
      #if SIFT

      if(cycle<3)
      {
        // Estimate normal, find keypoints and features ------------------ 
        std::cout<<"Started to compute scene cloud normals."<<std::endl;
        scene_cloud_normal_ptr = estimateNormals(scene_cloud_ptr,normal_radius_);
        scene_SIFT_keypoints_ptr = estimateKeypointsSIFT(scene_cloud_normal_ptr);
        scene_fpfh_ptr = fpfhExtractor(scene_cloud_ptr,scene_SIFT_keypoints_ptr,scene_cloud_normal_ptr,fpfh_radius_);
        // Find Correspondencses -----------------------------------------
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> feature_matching_est;
        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
        feature_matching_est.setInputSource(scene_fpfh_ptr);
        feature_matching_est.setInputTarget(ref_fpfh_ptr);
        feature_matching_est.determineCorrespondences(*correspondences);
        std::cout<<"Size of correspondences = "<< correspondences->size()<<std::endl;
        // RANSAC false matching removal  --------------------------------
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector_sac;
        pcl::CorrespondencesPtr correspondences_filtered(new pcl::Correspondences());
        rejector_sac.setInputSource(scene_SIFT_keypoints_ptr);
        rejector_sac.setInputTarget(ref_SIFT_keypoints_ptr);
        rejector_sac.setInlierThreshold(1);
        rejector_sac.setMaximumIterations(1000);
        rejector_sac.setRefineModel(true);    
        rejector_sac.setInputCorrespondences(correspondences);
        rejector_sac.getCorrespondences(*correspondences_filtered);
        std::cout<<"Size of correspondences after outlier correspondence removal = "<< correspondences_filtered->size()<<std::endl;
        // Motion Estimation using SVD -----------------------------------     
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> rough_T_est;
        rough_T_est.estimateRigidTransformation(*scene_SIFT_keypoints_ptr, *ref_SIFT_keypoints_ptr, *correspondences_filtered, T_matrix); 
        
      } 
       
      pcl::transformPointCloud(*scene_cloud_ptr, *scene_cloud_transformed_dl, T_matrix); 
      // Save the scene after initial transformation
      pcl::io::savePCDFileASCII("data_dl.pcd", *scene_cloud_transformed_dl);
      std::cout << "Rought Transformation coming from Pose Estimation" << std::endl << T_matrix << std::endl;   
      #endif
      // First Projection with Rough Transformation Matrix     
      pcl::transformPointCloud(*scene_cloud_ptr, *scene_cloud_transformed_dl, Rough_T_matrix_from_pose_est.inverse()); 
      // Save the scene after initial transformation
      pcl::io::savePCDFileASCII("data_dl.pcd", *scene_cloud_transformed_dl);
      std::cout << "Rought Transformation coming from Pose Estimation" << std::endl << Rough_T_matrix_from_pose_est.inverse() << std::endl;

        
      
      // Filling source cloud vector 
      if(cycle<3) // for initial iteration
      {
        // Converting filtered scene cloud to glm cloud for initial transformation
        glm_scene = pcl_to_glmvec(scene_cloud_transformed_dl);
      }
      else
      {
        // Converting filtered scene cloud to glm cloud   
        glm_scene = pcl_to_glmvec(scene_cloud_transformed_cuda);
      }
      #if GPU_KDTREE
        int sz = (int)log2(size); 
        std::vector<glm::ivec3> tk(glm_scene.size()*sz, glm::ivec3(0,0,0));
        track = &tk[0];   
        KDTree::initCpuKDTree(glm_ref, glm_ref_Tree);
        YbufferTree = &glm_ref_Tree[0];
        p_treesize = size;        
        p_tracksize = glm_scene.size()*sz;
      #endif
      // Iterative Closest Point Algorithm -----------------------------
      // Initialize and run ICP CUDA implementation 
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

      if(cycle<3) // for initial iteration
      {
        pcl::transformPointCloud(*scene_cloud_transformed_dl, *scene_cloud_transformed_cuda, icp_cuda_transformation); 
        // Save first iteration of ICP CUDA
        pcl::io::savePCDFileASCII("data_cuda_first_iteration.pcd", *scene_cloud_transformed_cuda);
      }
      else //
      {
        pcl::transformPointCloud(*scene_cloud_transformed_cuda, *scene_cloud_transformed_cuda, icp_cuda_transformation);
        // Save multiple iteration of ICP CUDA
        pcl::io::savePCDFileASCII("data_cuda_next_iterations.pcd", *scene_cloud_transformed_cuda);
      }     
     
       

      // Transform reference model into scene points using general transformation inverse
      if(cycle<3) //for initial iteration
      {

        gen_transformation = icp_cuda_transformation * Rough_T_matrix_from_pose_est.inverse();
        pcl::transformPointCloud (*ref_main_cloud_ptr, *final_output, gen_transformation.inverse());
      }
      else
      {
        pcl::transformPointCloud (*final_output, *final_output, icp_cuda_transformation.inverse()); 
      }     

      // Save model alignment to the scene
      pcl::io::savePCDFileASCII("data_body_alignment_to_scene.pcd", *final_output);
    
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
   
  
    pcl::toROSMsg(*final_output,output);
    output.header.frame_id = "camera_depth_optical_frame";
    output.header.stamp = ros::Time::now();
    pub.publish (output);    

    

    ros::spinOnce();
    r.sleep();
    
  }
  
  return 0;
}
