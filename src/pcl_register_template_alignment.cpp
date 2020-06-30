#include <vector>
#include <chrono>
#include <string>
#include <thread>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ros/ros.h>
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
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_macros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>

// Global Variables
const std::string PCD_PATH_REF = "PATH_TO_FULL_MODEL/body_visible.pcd";
const std::string PCD_PATH_SCENE = "PATH_TO_SAVED_SCENE_MODEL/scene_data.pcd";
pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_transformed (new pcl::PointCloud<pcl::PointXYZ>);
pcl::CorrespondencesPtr correspondences_filtered (new pcl::Correspondences());
pcl::CorrespondencesPtr correspondences (new pcl::Correspondences());
Eigen::Matrix4f T_matrix = Eigen::Matrix4f::Identity();

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


pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


pcl::visualization::PCLVisualizer::Ptr customColourVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


pcl::visualization::PCLVisualizer::Ptr normalsVis (
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


pcl::visualization::PCLVisualizer::Ptr shapesVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  //------------------------------------
  //-----Add shapes at cloud points-----
  //------------------------------------
  viewer->addLine<pcl::PointXYZRGB> (cloud->points[0],
                                     cloud->points[cloud->size() - 1], "line");
  viewer->addSphere (cloud->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");

  //---------------------------------------
  //-----Add shapes at other locations-----
  //---------------------------------------
  pcl::ModelCoefficients coeffs;
  coeffs.values.push_back (0.0);
  coeffs.values.push_back (0.0);
  coeffs.values.push_back (1.0);
  coeffs.values.push_back (0.0);
  viewer->addPlane (coeffs, "plane");
  coeffs.values.clear ();
  coeffs.values.push_back (0.3);
  coeffs.values.push_back (0.3);
  coeffs.values.push_back (0.0);
  coeffs.values.push_back (0.0);
  coeffs.values.push_back (1.0);
  coeffs.values.push_back (0.0);
  coeffs.values.push_back (5.0);
  viewer->addCone (coeffs, "cone");

  return (viewer);
}



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

pcl::visualization::PCLVisualizer::Ptr interactionCustomizationVis ()
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);

  viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)viewer.get ());
  viewer->registerMouseCallback (mouseEventOccurred, (void*)viewer.get ());

  return (viewer);
}

class FeatureCloud
{
  public:
    // A bit of shorthand
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
    typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
    typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

    FeatureCloud () :
      search_method_xyz_ (new SearchMethod),
      normal_radius_ (0.02f),
      feature_radius_ (0.02f)
    {}

    ~FeatureCloud () {}

    // Process the given cloud
    void
    setInputCloud (PointCloud::Ptr xyz)
    {
      xyz_ = xyz;
      filterInput();
      processInput();
    }

    // Load and process the cloud in the given PCD file
    void
    loadInputCloud (const std::string &pcd_file)
    {
      xyz_ = PointCloud::Ptr (new PointCloud);
      pcl::io::loadPCDFile (pcd_file, *xyz_);
      //filterInput();
      passthrough();
      processInput();
    }

    // Get a pointer to the cloud 3D points
    PointCloud::Ptr
    getPointCloud () const
    {
      return (xyz_);
    }

    // Get a pointer to the cloud of 3D surface normals
    SurfaceNormals::Ptr
    getSurfaceNormals () const
    {
      return (normals_);
    }

    // Get a pointer to the cloud of feature descriptors
    LocalFeatures::Ptr
    getLocalFeatures () const
    {
      return (features_);
    }

  protected:
  // Filter reference point cloud
    void filterInput()
    {   double leaf_size = 0.005f;
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (xyz_);
        sor.setLeafSize (leaf_size, leaf_size, leaf_size);
        sor.filter (*xyz_);
        // Remove Statistical Outliers -------------------------------------
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_statistic_removal;
        sor_statistic_removal.setInputCloud (xyz_);
        sor_statistic_removal.setMeanK (100);
        sor_statistic_removal.setStddevMulThresh(1.0);
        sor_statistic_removal.filter (*xyz_);
    }
    void passthrough()
    {
      const float depth_limit = 2;
      const float height_limit = 1.5;
      const float width_limit = 0.4;
      pcl::PassThrough<pcl::PointXYZ> pass;
      pass.setInputCloud (xyz_);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0, depth_limit);
      pass.filter (*xyz_);
      pass.setFilterFieldName("y");
      pass.setFilterLimits(0.5, height_limit);
      pass.setFilterLimitsNegative(true);
      pass.filter(*xyz_);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(-width_limit+0.15,width_limit-0.1);
      pass.setFilterLimitsNegative(false);
      pass.filter(*xyz_);
    }
    // Compute the surface normals and local features
    void
    processInput ()
    {
      computeSurfaceNormals ();
      computeLocalFeatures ();
    }

    // Compute the surface normals
    void
    computeSurfaceNormals ()
    {
      normals_ = SurfaceNormals::Ptr (new SurfaceNormals);

      pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
      norm_est.setInputCloud (xyz_);
      norm_est.setRadiusSearch(normal_radius_);
      norm_est.setSearchMethod (search_method_xyz_);
      norm_est.compute (*normals_);
    }

    // Compute the local feature descriptors
    void
    computeLocalFeatures ()
    {
      features_ = LocalFeatures::Ptr (new LocalFeatures);

      pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
      fpfh_est.setInputCloud (xyz_);
      fpfh_est.setInputNormals (normals_);
      fpfh_est.setSearchMethod (search_method_xyz_);
      fpfh_est.setRadiusSearch (feature_radius_);
      fpfh_est.compute (*features_);
    }

  private:
    // Point cloud data
    PointCloud::Ptr xyz_;
    SurfaceNormals::Ptr normals_;
    LocalFeatures::Ptr features_;
    SearchMethod::Ptr search_method_xyz_;

    // Parameters
    float normal_radius_;
    float feature_radius_;
};

class trEstimation
{

  private:

  FeatureCloud target_; //ref
  FeatureCloud template_; //scene
  pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33,pcl::FPFHSignature33> feature_matching_est;
  pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector_sac;

  public:


    // Set the given cloud as the target to which the templates will be aligned
    void
    findCorrespondence (FeatureCloud &target_cloud, FeatureCloud &template_cloud)
    {

      target_ = target_cloud;
      template_ = template_cloud;
      feature_matching_est.setInputSource(template_cloud.getLocalFeatures());
      feature_matching_est.setInputTarget(target_cloud.getLocalFeatures());
      feature_matching_est.determineCorrespondences(*correspondences);
      std::cout<<"Size of correspondences = "<< correspondences->size()<<std::endl;
    }
    //Reject outlier correspondences
    void
    rejectOutlier(FeatureCloud &target_cloud, FeatureCloud &template_cloud)
    {

      rejector_sac.setInputSource(template_.getPointCloud());
      rejector_sac.setInputTarget(target_.getPointCloud());
      rejector_sac.setInlierThreshold(3);
      rejector_sac.setMaximumIterations(500);
      rejector_sac.setRefineModel(true);
      rejector_sac.setInputCorrespondences(correspondences);
      rejector_sac.getCorrespondences(*correspondences_filtered);
      std::cout<<"Size of correspondences after outlier correspondence removal = "<< correspondences_filtered->size()<<std::endl;
    }
    void estimateTransformationMatrix()
    {
       //T_matrix = rejector_sac.getBestTransformation();
       pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> rough_T_est;
       rough_T_est.estimateRigidTransformation(*template_.getPointCloud(),*target_.getPointCloud(), *correspondences_filtered, T_matrix);
       std::cout<<T_matrix<<std::endl;
    }

};
int main (int argc, char** argv)
{

    int cycle = 0;
    // Initialize ROS
    ros::init (argc, argv, "point_cloud_alignment");
    ros::NodeHandle nh;
    ros::Rate r(5);
    // ROS publisher
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2> ("registered_points", 1);

    // Import scene point cloud data
    std::vector<FeatureCloud> object_templates;
    object_templates.resize (0);
    FeatureCloud template_cloud;
    template_cloud.loadInputCloud(PCD_PATH_SCENE);
    object_templates.push_back(template_cloud);

    // To show scene point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_filtered_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    scene_filtered_ptr = template_cloud.getPointCloud();

    // Load ref cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (PCD_PATH_REF, *ref_cloud_ptr) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    FeatureCloud target_cloud;
    target_cloud.setInputCloud(ref_cloud_ptr);
    trEstimation trans;
    trans.findCorrespondence(target_cloud,template_cloud);
    trans.rejectOutlier(target_cloud,template_cloud);
    trans.estimateTransformationMatrix();


    //  (1) Save the aligned template for visualization -------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*template_cloud.getPointCloud(), *transformed_cloud, T_matrix);
    pcl::io::savePCDFileBinary ("rough_aligned_cloud.pcd", *transformed_cloud);

    //  (3) Iterative Closest Point Algorithm ----------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_output(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setInputSource(transformed_cloud);
    icp.setInputTarget(ref_cloud_ptr);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaximumIterations(100000);
    icp.align(*final_output);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation()<< std::endl;
    pcl::io::savePCDFileASCII("icp_aligned_cloud.pcd", *final_output);

    // Transform reference model into scene points using general transformation inverse
    Eigen::Matrix4f icp_transformation = icp.getFinalTransformation();
    Eigen::Matrix4f gen_transformation = icp_transformation*T_matrix;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_ref_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*target_cloud.getPointCloud(), *aligned_ref_cloud, gen_transformation.inverse());
    pcl::io::savePCDFileBinary ("aligned_ref_output.pcd", *aligned_ref_cloud);


    // ROS Message
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*aligned_ref_cloud,output);
    output.header.frame_id = "xtion_depth_optical_frame";
    // To visualize PointCloud
    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis(scene_filtered_ptr);
    while(ros::ok())
    {
      cycle++;
      pub.publish(output);
      ros::spinOnce();
      r.sleep();
      if(!viewer->wasStopped())
      {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    return 0;
}
