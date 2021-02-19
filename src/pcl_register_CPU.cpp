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
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/StdVector>


pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_main_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr final_output(new pcl::PointCloud<pcl::PointXYZ>);
// Reference to Rough Transformation matrix coming from Pose Estimation using DL
Eigen::Matrix4f Rough_T_matrix_from_pose_est = Eigen::Matrix4f::Identity();

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

int main (int argc, char** argv)
{ 

  int cycle = 0;
  const std::string PCD_PATH = "/home/dlar/catkin_ws/src/pcl_registration/Data/body_visible_rotated.pcd";
  const std::string PCD_MAIN_PATH = "/home/dlar/catkin_ws/src/pcl_registration/Data/body_visible_rotated.pcd";
  
  // Initialize ROS
  ros::init (argc, argv, "point_cloud_register_using_detection");
  ros::NodeHandle nh;
  ros::Rate r(20);
  
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

  // Filtering Reference Cloud ---------------------------------------
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
  ref_cloud_normal_ptr = estimateNormals(ref_cloud_filtered_ptr,normal_radius_);
  ref_SIFT_keypoints_ptr = estimateKeypointsSIFT(ref_cloud_normal_ptr);
  ref_fpfh_ptr = fpfhExtractor(ref_cloud_filtered_ptr,ref_SIFT_keypoints_ptr,ref_cloud_normal_ptr,fpfh_radius_);
  // Filter parameters
  float depth_limit;
  float height_limit;
  float width_limit;
  
  // Rough Transformation matrix  
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed_dl(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_ref_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  std::cout<<"ROS Cycle starts"<<std::endl; 
  
  while(ros::ok())
  { 
    cycle++;
    
    if(cycle>1)   
    {                 
      // First filter scene cloud data ---------------------------------
      // ...removing distant points
      pcl::PassThrough<pcl::PointXYZ> pass;
      pass.setInputCloud (scene_cloud_ptr);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0, Rough_T_matrix_from_pose_est(2,3)+0.10);
      pass.filter (*scene_cloud_ptr);         
      pass.setFilterFieldName("x");
      width_limit = Rough_T_matrix_from_pose_est(0,3);
      pass.setFilterLimits(width_limit-0.40,width_limit+0.40);
      pass.filter(*scene_cloud_ptr);  
      pass.setFilterFieldName("y");
      height_limit = Rough_T_matrix_from_pose_est(1,3);
      pass.setFilterLimits(height_limit-0.10, height_limit+0.75);
      pass.filter(*scene_cloud_ptr);
      // Downsampling scene point cloud
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud (scene_cloud_ptr);
      sor.setLeafSize (0.005, 0.005, 0.005);
      sor.filter (*scene_cloud_ptr);
        
       
      // Remove Statistical Outliers -------------------------------------
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_statistic_removal;
      sor_statistic_removal.setInputCloud (scene_cloud_ptr);
      sor_statistic_removal.setMeanK (50);
      sor_statistic_removal.setStddevMulThresh(1.0);
      sor_statistic_removal.filter (*scene_cloud_ptr);   
      pcl::io::savePCDFileASCII("data_scene.pcd", *scene_cloud_ptr);
               
      // First Projection with Rough Transformation Matrix 
      if(cycle<3)
      {
      pcl::transformPointCloud(*scene_cloud_ptr, *scene_cloud_transformed_dl, Rough_T_matrix_from_pose_est.inverse()); 
      pcl::io::savePCDFileASCII("data_dl.pcd", *scene_cloud_transformed_dl);
      std::cout << "Rought Transformation coming from Pose Estimation" << std::endl << Rough_T_matrix_from_pose_est.inverse() << std::endl; 
      }
      
      auto start = std::chrono::high_resolution_clock::now(); // To compute time interval ICP computation needs
      // Iterative Closest Point Algorithm -----------------------------    
      pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
      icp.setInputSource(scene_cloud_transformed_dl);
      icp.setInputTarget(ref_cloud_filtered_ptr);
      icp.setTransformationEpsilon(1e-8);
      icp.setMaximumIterations(100000);
      icp.align(*scene_cloud_transformed_dl);
      auto end = std::chrono::high_resolution_clock::now();    
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      std::cout << icp.getFinalTransformation()<< std::endl;
      pcl::io::savePCDFileASCII("data_icp_CPU.pcd", *scene_cloud_transformed_dl);
      
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
	    std::cout << "duration: " << duration.count() << std::endl;

      Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();
      if(cycle<3)
      {
      // Transform reference model into scene points using general transformation inverse
      Eigen::Matrix4f gen_transformation = icp_transformation*Rough_T_matrix_from_pose_est.inverse();
      pcl::transformPointCloud (*ref_cloud_ptr, *final_output, gen_transformation.inverse());   
      }  
      else
      {
        pcl::transformPointCloud (*final_output, *final_output, icp_transformation.inverse());
      } 
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
