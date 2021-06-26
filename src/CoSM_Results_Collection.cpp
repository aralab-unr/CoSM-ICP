#include <iostream>
#include <string>
#include<vector>
#include <fstream>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc
#include <pcl/registration/transformation_estimation_correntropy_svd.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/features/normal_3d.h>



#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <random>
#include<math.h>


#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/pyramid_feature_matching.h>
#include <pcl/features/ppf.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/filters/voxel_grid.h>
// We need Histogram<2> to function, so we'll explicitly add kdtree_flann.hpp here
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <Eigen/Core>

using namespace pcl;
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointTN;
typedef pcl::Normal PointN;

typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointTN> PointCloudTN;
typedef pcl::PointCloud<PointN> PointCloudN;

bool next_iteration = false;

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az)
 {
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

void print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}


void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event,
                       void* nothing)
{
  if (event.getKeySym () == "space" && event.keyDown ())
    next_iteration = true;
}
void
loadFile(const char* fileName,
   pcl::PointCloud<pcl::PointXYZ> &cloud
)
{
  pcl::PolygonMesh mesh;
  
  if ( pcl::io::loadPolygonFile ( fileName, mesh ) == -1 )
  {
    PCL_ERROR ( "loadFile faild." );
    return;
  }
  else
    pcl::fromPCLPointCloud2<pcl::PointXYZ> ( mesh.cloud, cloud );
  
  // remove points having values of nan
  std::vector<int> index;
  pcl::removeNaNFromPointCloud ( cloud, cloud, index );
}

int
main (int argc,
      char* argv[])
{
 // The point clouds we will be using
  PointCloudT::Ptr cloud_target (new PointCloudT);  // Original point cloud
  PointCloudT::Ptr cloud_tr (new PointCloudT);  // Transformed point cloud
  PointCloudT::Ptr cloud_source (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source1 (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_source2 (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_target1 (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target2 (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_st_svd (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_st_svd (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_p2pl (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_p2pl (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_p2pl_weighted (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_p2pl_weighted (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_p2pl_lls (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_p2pl_lls (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_p2pl_lls_weighted (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_p2pl_lls_weighted (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_gicp (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_gicp (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_icp_nl (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_icp_nl (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_ndt (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_ndt (new PointCloudT);  // ICP output point cloud

  PointCloudT::Ptr cloud_source_cor_svd (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_target_cor_svd (new PointCloudT);  // ICP output point cloud

  PointCloudN::Ptr cloud_source_normal (new PointCloudN);  // ICP output point cloud
  PointCloudN::Ptr cloud_target_normal (new PointCloudN);  // ICP output point cloud

  PointCloudTN::Ptr cloud_source_with_normals_p2pl (new PointCloudTN);  // ICP output point cloud
  PointCloudTN::Ptr cloud_target_with_normals_p2pl (new PointCloudTN);  // ICP output point cloud

  PointCloudTN::Ptr cloud_source_with_normals_p2pl_weighted (new PointCloudTN);  // ICP output point cloud
  PointCloudTN::Ptr cloud_target_with_normals_p2pl_weighted (new PointCloudTN);  // ICP output point cloud

  PointCloudTN::Ptr cloud_source_with_normals_p2pl_lls (new PointCloudTN);  // ICP output point cloud
  PointCloudTN::Ptr cloud_target_with_normals_p2pl_lls (new PointCloudTN);  // ICP output point cloud

  PointCloudTN::Ptr cloud_source_with_normals_p2pl_lls_weighted (new PointCloudTN);  // ICP output point cloud
  PointCloudTN::Ptr cloud_target_with_normals_p2pl_lls_weighted (new PointCloudTN);  // ICP output point cloud

  PointCloudTN::Ptr cloud_aligned_normal1 (new PointCloudTN);  // ICP output point cloud
  
  PointCloudT::Ptr cloud_aligned1 (new PointCloudT);  // ICP output point cloud
  PointCloudT::Ptr cloud_aligned2(new PointCloudT);  // ICP output point cloud

  // Checking program arguments
  if (argc < 2)
  {
    printf ("Usage :\n");
    printf ("\t\t%s file.ply number_of_ICP_iterations\n", argv[0]);
    PCL_ERROR ("Provide one ply file.\n");
    return (-1);
  }

  int iterations = 1;  // Default number of ICP iterations
  int iter1=iterations;
  if (argc > 2)
  {
    // If the user passed the number of iteration as an argument
    iter1 = atoi (argv[2]);
    if (iter1 < 0)
    {
      PCL_ERROR ("Number of initial iterations must be >= 1\n");
      return (-1);
    }
  }
  iterations=iter1;
  
  float leaf_s=0.01f;
  if (argc > 3)
  {
    // If the user passed the number of iteration as an argument
    leaf_s = atof (argv[3]);
  }
  std::cout<<"\nleafsize="<<leaf_s<<" iterations="<<iterations;

  pcl::console::TicToc time;
  time.tic ();
  loadFile(argv[1],*cloud_target);
  // if (pcl::io::loadPLYFile (argv[1], *cloud_target) < 0)
  // {
  //   PCL_ERROR ("Error loading cloud %s.\n", argv[1]);
  //   return (-1);
  // }
  // std::vector<int> index;
  // pcl::removeNaNFromPointCloud (*cloud_target,*cloud_target, index );
  std::cout << "\nLoaded file " << argv[1] << " (" << cloud_target->size () << " points) in " << time.toc () << " ms\n" << std::endl;

   
   pcl::VoxelGrid<PointT> sor;
   sor.setLeafSize (leaf_s, leaf_s, leaf_s);

   sor.setInputCloud (cloud_target);
  
  sor.filter (*cloud_target);

  

  std::cout<<"\nTarget size after voxelgrid="<<cloud_target->size(); 
 

  //std::string path1="/media/aralab/DATA/ashu_ws/CICP-SLAM-master/mcc-icp/MCC_ICP_PCL/test_results/dragon_dataset/";
  std::string path(argv[4]);
  std::cout<<"\nPath1="<<path;
  std::string method="ICP_STANDARD_SVD";
  

  ofstream fs1;
  fs1.open(path+"initial_settings.txt",ios::app);



  // Defining a rotation matrix and translation vector
  ////Define a simulated model/////////////////Comment later on to use from file///////////////////
  // int32_t num = 100;
  // double min1=-2.0,max1=2.0;
  // double factor=(max1-min1)/sqrt(num);
  
  // std::cout<<"\ntarget_size="<<cloud_target->width<<" height= "<<cloud_target->height;
  // cloud_target->clear();
  // int k=0;
  //  cloud_target->width=num;
  // cloud_target->height=1;
  // for (double x=min1; x<max1; x+=factor) 
  // {
  //   for (double y=min1; y<max1; y+=factor) 
  //   {
  //     double z=5*x*exp(-x*x-y*y);
  //     pcl::PointXYZ np;
  //     np.x=x;
  //     np.y=y;
  //     np.z=z;

  //     cloud_target->points.push_back(np);
  //     k++;
  //   }
  // }
  // //cloud_target->width=k;
  // //cloud_target->height=1;
  
  // std::cout<<"\ntarget_size2="<<cloud_target->width<<" height= "<<cloud_target->height;

  /////////////////////////////////////////////////////////////////////////////////////////////////

  ////Generate random for checking under various rotation and trasnlation///////

  double std_dev_tx=10;
  double std_dev_ty=10;
  double std_dev_tz=10;

  double noise_std_dev_tx=0.05;
  double noise_std_dev_ty=0.05;
  double noise_std_dev_tz=0.05;

  double min_angle=-6.28319;
  double max_angle=6.28319;

  double min_translation_x=-1*std_dev_tx;
  double min_translation_y=-1*std_dev_ty;
  double min_translation_z=-1*std_dev_tz;

  double max_translation_x=std_dev_tx;
  double max_translation_y=std_dev_ty;
  double max_translation_z=std_dev_tz;

  double min_noise_translation_x=-1*noise_std_dev_tx;
  double min_noise_translation_y=-1*noise_std_dev_ty;
  double min_noise_translation_z=-1*noise_std_dev_tz;

  double max_noise_translation_x=noise_std_dev_tx;
  double max_noise_translation_y=noise_std_dev_tx;
  double max_noise_translation_z=noise_std_dev_tx;

  
  fs1<<"\nstd_dev_tx="<<std_dev_tx<<"\nstd_dev_ty="<<std_dev_ty<<"\nstd_dev_tz="<<std_dev_tz;
  fs1<<"\nnoise_std_dev_tx="<< noise_std_dev_tx<<"\nnoise_std_dev_ty="<<noise_std_dev_ty<<"\nnoise_std_dev_tz="<< noise_std_dev_tz;



  unsigned seed_ax = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ax(seed_ax);

  std::normal_distribution<double> distribution_ax(0,max_angle);

   unsigned seed_ay = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ay(seed_ay);

  std::normal_distribution<double> distribution_ay(0,max_angle);

  unsigned seed_az = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_az(seed_az);

  std::normal_distribution<double> distribution_az(0,max_angle);
   std::cout<<"\nMaxvalue="<<distribution_az.max();



  unsigned seed_tx = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_tx(seed_tx);

  std::normal_distribution<double> distribution_tx(0,std_dev_tx);

   unsigned seed_ty = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ty(seed_ty);

  std::normal_distribution<double> distribution_ty(0,std_dev_tx);

  unsigned seed_tz = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_tz(seed_tz);

  std::normal_distribution<double> distribution_tz(0,std_dev_tx);


  unsigned seed_x = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_x(seed_x);

  std::normal_distribution<double> distribution_x(0,0.05);

  unsigned seed_y = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_y(seed_y);

  std::normal_distribution<double> distribution_y(0,0.05);

  unsigned seed_z = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_z(seed_z);

  std::normal_distribution<double> distribution_z(0,0.05);

 
  for(int l=0;l<5;l++)
  {
    iterations=0;
   ofstream fs2;
   std::stringstream ss1;
   ss1<<l;
   std::string ctr = ss1.str ();
   fs2.open(path+"initial_settings_"+ctr+".txt",ios::app);


  double ax1=distribution_ax(generator_ax);
  double ay1=distribution_ay(generator_ay);
  double az1=distribution_az(generator_az);

  double tx1=distribution_tx(generator_tx);
  double ty1=distribution_ty(generator_ty);
  double tz1=distribution_tz(generator_tz);


  
  while(ax1>max_angle || ax1<min_angle)
  {
      ax1=distribution_ax(generator_ax);
    
  }
  std::cout<<"\nax1="<<ax1;

  while(ay1>max_angle || ay1<min_angle)
  {
      ay1=distribution_ay(generator_ay);
  }

  while(az1>max_angle || az1<min_angle)
  {
      az1=distribution_az(generator_az);
  }


  while(tx1>max_translation_x || tx1<min_translation_x)
  {
         tx1=distribution_tx(generator_tx);
  }

  while(ty1>max_translation_y || ty1<min_translation_y)
  {
      ty1=distribution_ty(generator_ty);
  }

  while(tz1>max_translation_z || tz1<min_translation_z)
  {
      tz1=distribution_tz(generator_tz);
  }

  fs2<<"\nRandom Transformation Angle(Radians(x,y,z))={"<<ax1<<","<<ay1<<","<<az1<<"}";

  

  
  

  
  Eigen::Affine3d r = create_rotation_matrix(ax1,ay1,az1);
 
  Eigen::Affine3d tr(Eigen::Translation3d(Eigen::Vector3d(tx1,ty1,tz1)));

  Eigen::Matrix4d m = (tr * r).matrix(); // Option 1

  m = tr.matrix(); // Option 2
  m *= r.matrix();

  Eigen::Matrix3d mr=m.topLeftCorner(3,3);
  Eigen::Vector3d euler_angles = mr.eulerAngles(2,1,0);
  fs2<<"\nRandom Transformation Angle(Degrees(x,y,z))={"<<euler_angles.transpose()<<"}";
  fs2<<"\nRandom Transformation translation(x,y,z)={"<<tx1<<","<<ty1<<","<<tz1<<"}\n";

  fs2<<"\nTransformation Matrix(from source to target)=\n"<<m<<"\n";


  for(int t1=0;t1<4;t1++)
  {
    for(int t2=0;t2<4;t2++)
    {
     fs2<<m(t1,t2)<<" ";
    }
  }
  fs2.close();

  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
  Eigen::Matrix4d transformation_matrix1 = Eigen::Matrix4d::Identity ();

  // A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  double theta = M_PI/10;  // The angle of rotation in radians
  transformation_matrix (0, 0) = std::cos (theta);
  transformation_matrix (0, 1) = -sin (theta);
  transformation_matrix (1, 0) = sin (theta);
  transformation_matrix (1, 1) = std::cos (theta);

  // A translation on Z axis (0.4 meters)
  transformation_matrix (2, 3) = 0.05;

  //Create your own Transformation matrix for evaluation. Comment this if you intend to use the 
  //randomly generated transformation matrix
  // m(0,0)= -0.4862182;   m(0,1)= -0.0018080;    m(0,2)= -0.8738356;    m(0,3)= -0.874;
  // m(1,0)= 0.7801269;    m(1,1)= -0.4514303;    m(1,2)= -0.4331429;    m(1,3)= -0.433;
  // m(2,0)= -0.3936927;     m(2,1)= -0.8923045;    m(2,2)= 0.2209041;   m(2,3)= 0.221;
  // m(3,0)=0;             m(3,1)=0;              m(3,2)=0;              m(3,3)=1;

  // Display in terminal the transformation matrix
  std::cout << "\nApplying this rigid transformation to: cloud_target -> cloud_source" << std::endl;
  print4x4Matrix (m);
   
  // Executing the transformation
  pcl::transformPointCloud (*cloud_target, *cloud_source,m);


   //............adding random attack vectors to the source dataset............

  /////////////////////// Add outliers (at random locations with random values) /////////////////////////
  double per=50.0;
  int nop=(per*cloud_target->size())/100;



  
 //ofstream ofs1,ofs2;

  //ofs2.open("iter_data/idx_att.txt",ios::app);
  for(int i=0;i<nop;i++)
  {
     int v1=rand()%cloud_target->size();
    // ofs2<<v1/3<<"\n";
     double vx=distribution_x(generator_x);
     double vy=distribution_y(generator_y);
     double vz=distribution_z(generator_z);


  while(vx>max_noise_translation_x || vx<min_noise_translation_x)
  {
         vx=distribution_x(generator_x);
  }

  while(vy>max_noise_translation_y || vy<min_noise_translation_y)
  {
      vy=distribution_y(generator_y);
  }

  while(vz>max_noise_translation_z || vz<min_noise_translation_z)
  {
      vz=distribution_z(generator_z);
  }

     std::cout<<"\nvx="<<vx<<" vy="<<vy<<" vz="<<vz;

     cloud_source->points[v1].x+=vx;
     cloud_source->points[v1].y+=vy;
     cloud_source->points[v1].z+=vz;
    // T[v1]=T[v1]+v2;
  }
  //ofs2.close();
  //////////////////////////////////////////////////////////
   // const int nol=0.00*cloud_source->points.size();
   // std::cout<<"\nnol="<<nol;
   // //int aloc[nol];
   // double avals[3]={0.04,0.04,0.04};
   // const int sz=cloud_source->points.size();
   // int srcflags[sz];
   // for (int i=0;i<sz;i++)srcflags[i]=0;
   // for (int i=0;i<nol;i++)
   // { 
   //   int v1=rand()%cloud_source->points.size();
   //   //std:cout<<"\nv1="<<v1;
   //   double N=0.1;
   //   double M=0.5;
 
   //   double val_x= M + (rand()/( RAND_MAX / (N-M)));
   //   double val_y= M + (rand()/( RAND_MAX / (N-M)));
   //   double val_z= M + (rand()/( RAND_MAX / (N-M)));
   //   if(srcflags[v1]==0)
   //   {
   //   cloud_source->points[v1].x+=val_x;
   //   cloud_source->points[v1].y+=val_y;
   //   cloud_source->points[v1].z+=val_z;
   //   srcflags[v1]=1;
   //   } 
   // }
   
   //////save the source and the target in a file //////////////////////////////
  ofstream ofs1,ofs2;
  ofs1.open("source_points.txt",ios::app);
  ofs2.open("target_points.txt",ios::app);
  
   ofs1<<cloud_source->size();
   ofs2<<cloud_target->size();

  for(int i=0;i<cloud_source->size();i++)
  {
    ofs1<<"\n"<<cloud_source->points[i].x<<" "<<cloud_source->points[i].y<<" "<<cloud_source->points[i].z; 
    ofs2<<"\n"<<cloud_target->points[i].x<<" "<<cloud_target->points[i].y<<" "<<cloud_target->points[i].z; 

  } 
  ofs1.close();
  ofs2.close();
  /////////////////////////////////////////////////////////////////////////////
 
   /////////////////////////////////////////////////////////////////////////////
  *cloud_tr = *cloud_source;  // We backup cloud_source into cloud_tr for later use
  *cloud_source1=*cloud_source;
  *cloud_source2=*cloud_source;

   *cloud_target1=*cloud_target;

  pcl::copyPointCloud(*cloud_source,*cloud_source_st_svd);
  pcl::copyPointCloud(*cloud_target,*cloud_target_st_svd);
  
  pcl::copyPointCloud(*cloud_source,*cloud_source_cor_svd);
  pcl::copyPointCloud(*cloud_target,*cloud_target_cor_svd);

  pcl::copyPointCloud(*cloud_source,*cloud_source_p2pl);
  pcl::copyPointCloud(*cloud_target,*cloud_target_p2pl);

  pcl::copyPointCloud(*cloud_source,*cloud_source_p2pl_weighted);
  pcl::copyPointCloud(*cloud_target,*cloud_target_p2pl_weighted);

  pcl::copyPointCloud(*cloud_source,*cloud_source_p2pl_lls);
  pcl::copyPointCloud(*cloud_target,*cloud_target_p2pl_lls);

  pcl::copyPointCloud(*cloud_source,*cloud_source_p2pl_lls_weighted);
  pcl::copyPointCloud(*cloud_target,*cloud_target_p2pl_lls_weighted);

  pcl::copyPointCloud(*cloud_source,*cloud_source_gicp);
  pcl::copyPointCloud(*cloud_target,*cloud_target_gicp);

  pcl::copyPointCloud(*cloud_source,*cloud_source_icp_nl);
  pcl::copyPointCloud(*cloud_target,*cloud_target_icp_nl);

  pcl::copyPointCloud(*cloud_source,*cloud_source_ndt);
  pcl::copyPointCloud(*cloud_target,*cloud_target_ndt);
time.tic ();

 /////////////Compute normals for point ot plane and other methods/////////////////////////////////
 pcl::NormalEstimation<PointT, PointN> norm_est;
 search::KdTree<PointT>::Ptr search_tree1 (new search::KdTree<PointT> ());
 norm_est.setSearchMethod (search_tree1);
 norm_est.setRadiusSearch (2.0);
 norm_est.setInputCloud (cloud_source);
 norm_est.compute (*cloud_source_normal);
  
 pcl::NormalEstimation<PointT, PointN> norm_est2;
 search::KdTree<PointT>::Ptr search_tree2 (new search::KdTree<PointT> ());
 norm_est2.setSearchMethod (search_tree2);
 norm_est2.setRadiusSearch (2.0);
 norm_est2.setInputCloud (cloud_target);
 norm_est2.compute (*cloud_target_normal);
//////////////////////////////////////////////////////////////////////////////////////////////////



  
///////////////////// // The Iterative Closest Point algorithm (Standard SVD)//////////////////////////////////
 
  
 pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp_st_svd ( new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> () );
 pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, PointXYZ>::Ptr trans_svd (new pcl::registration::TransformationEstimationSVD<PointXYZ, PointXYZ>);
 icp_st_svd->setMaximumIterations (iterations);
 icp_st_svd->setTransformationEstimation (trans_svd);
 icp_st_svd->setInputSource ( cloud_source_st_svd); // not cloud_source, but cloud_source_trans!
 icp_st_svd->setInputTarget ( cloud_target_st_svd );
//////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////The Iterative Closest Point algorithm (Point to Plane)////////////////////////////////////////
 pcl::IterativeClosestPoint<PointTN,PointTN>::Ptr icp_p2pl ( new pcl::IterativeClosestPoint<PointTN, PointTN> () );
 pcl::registration::TransformationEstimationPointToPlane<PointTN, PointTN>::Ptr trans_p2pl(new pcl::registration::TransformationEstimationPointToPlane<PointTN, PointTN>);
 std::cout<<"\nCloudsourceN="<<cloud_source_normal->size();
 std::cout<<"\nCloudtargetN="<<cloud_target_normal->size();
 pcl::concatenateFields (*cloud_target_p2pl, *cloud_target_normal, *cloud_target_with_normals_p2pl);
 pcl::concatenateFields (*cloud_source_p2pl, *cloud_source_normal, *cloud_source_with_normals_p2pl);
 
 icp_p2pl->setMaximumIterations (iterations);
 icp_p2pl->setTransformationEstimation (trans_p2pl);
 icp_p2pl->setInputSource ( cloud_source_with_normals_p2pl); // not cloud_source, but cloud_source_trans!
 icp_p2pl->setInputTarget ( cloud_target_with_normals_p2pl);

///////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////The Iterative Closest Point algorithm (Point to Plane Weighted)////////////////////////////////////////
 pcl::IterativeClosestPoint<PointTN,PointTN>::Ptr icp_p2pl_weighted ( new pcl::IterativeClosestPoint<PointTN, PointTN> () );
 pcl::registration::TransformationEstimationPointToPlaneWeighted<PointTN, PointTN>::Ptr trans_p2pl_weighted(new pcl::registration::TransformationEstimationPointToPlaneWeighted<PointTN, PointTN>);
 //std::cout<<"\nCloudsourceN="<<cloud_source_normal1->size();
 //std::cout<<"\nCloudtargetN="<<cloud_target_normal1->size();
 pcl::concatenateFields (*cloud_target_p2pl_weighted, *cloud_target_normal, *cloud_target_with_normals_p2pl_weighted);
 pcl::concatenateFields (*cloud_source_p2pl_weighted, *cloud_source_normal, *cloud_source_with_normals_p2pl_weighted);
 
 icp_p2pl_weighted->setMaximumIterations (iterations);
 icp_p2pl_weighted->setTransformationEstimation (trans_p2pl_weighted);
 icp_p2pl_weighted->setInputSource ( cloud_source_with_normals_p2pl_weighted); // not cloud_source, but cloud_source_trans!
 icp_p2pl_weighted->setInputTarget ( cloud_target_with_normals_p2pl_weighted);

/////////////////////////////////////////////////////////////////////////////////////////////////////////// 



///////////////////////The Iterative Closest Point algorithm (Point to Plane LLS)////////////////////////////////////////
 pcl::IterativeClosestPoint<PointTN,PointTN>::Ptr icp_p2pl_lls ( new pcl::IterativeClosestPoint<PointTN, PointTN> () );
 pcl::registration::TransformationEstimationPointToPlaneLLS<PointTN, PointTN>::Ptr trans_p2pl_lls(new pcl::registration::TransformationEstimationPointToPlaneLLS<PointTN, PointTN>);
 //std::cout<<"\nCloudsourceN="<<cloud_source_normal1->size();
 //std::cout<<"\nCloudtargetN="<<cloud_target_normal1->size();
 pcl::concatenateFields (*cloud_target_p2pl_lls, *cloud_target_normal, *cloud_target_with_normals_p2pl_lls);
 pcl::concatenateFields (*cloud_source_p2pl_lls, *cloud_source_normal, *cloud_source_with_normals_p2pl_lls);
 
 icp_p2pl_lls->setMaximumIterations (iterations);
 icp_p2pl_lls->setTransformationEstimation (trans_p2pl_lls);
 icp_p2pl_lls->setInputSource ( cloud_source_with_normals_p2pl_lls); // not cloud_source, but cloud_source_trans!
 icp_p2pl_lls->setInputTarget ( cloud_target_with_normals_p2pl_lls);

/////////////////////////////////////////////////////////////////////////////////////////////////////////// 



///////////////////////The Iterative Closest Point algorithm (Point to Plane LLS Weighted)////////////////////////////
 pcl::IterativeClosestPoint<PointTN,PointTN>::Ptr icp_p2pl_lls_weighted ( new pcl::IterativeClosestPoint<PointTN, PointTN> () );
 pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<PointTN, PointTN>::Ptr trans_p2pl_lls_weighted(new pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<PointTN, PointTN>);
 //std::cout<<"\nCloudsourceN="<<cloud_source_normal1->size();
 //std::cout<<"\nCloudtargetN="<<cloud_target_normal1->size();
 pcl::concatenateFields (*cloud_target_p2pl_lls_weighted, *cloud_target_normal, *cloud_target_with_normals_p2pl_lls_weighted);
 pcl::concatenateFields (*cloud_source_p2pl_lls_weighted, *cloud_source_normal, *cloud_source_with_normals_p2pl_lls_weighted);
 
 icp_p2pl_lls_weighted->setMaximumIterations (iterations);
 icp_p2pl_lls_weighted->setTransformationEstimation (trans_p2pl_lls_weighted);
 icp_p2pl_lls_weighted->setInputSource ( cloud_source_with_normals_p2pl_lls_weighted); // not cloud_source, but cloud_source_trans!
 icp_p2pl_lls_weighted->setInputTarget ( cloud_target_with_normals_p2pl_lls_weighted);

/////////////////////////////////////////////////////////////////////////////////////////////////////////// 


  
///////////////////// // Generalized ICP //////////////////////////////////
 
  
 pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr gicp ( new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> () );
 gicp->setMaximumIterations (iterations);
 gicp->setInputSource ( cloud_source_gicp); // not cloud_source, but cloud_source_trans!
 gicp->setInputTarget ( cloud_target_gicp );
//////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////// // The Iterative Closest Point algorithm (Non Linear)//////////////////////////////////
  pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp_nl ( new pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> () );
 icp_nl->setMaximumIterations (iterations);
 icp_nl->setInputSource ( cloud_source_icp_nl); // not cloud_source, but cloud_source_trans!
 icp_nl->setInputTarget ( cloud_target_icp_nl );
//////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////// // The Iterative Closest Point algorithm (Non Linear)//////////////////////////////////
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt ( new pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> () );
 ndt->setMaximumIterations (iterations);
 ndt->setInputSource ( cloud_source_ndt); // not cloud_source, but cloud_source_trans!
 ndt->setInputTarget ( cloud_target_ndt);
ndt->setStepSize (1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
   ndt->setResolution (4.5);
//////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////ICP CORRENTROPY SVD//////////////////////////////////////////////////////////////////
  
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp_cor ( new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> () );
  pcl::registration::TransformationEstimationCorrentropySVD<pcl::PointXYZ, PointXYZ>::Ptr trans_cor_svd (new pcl::registration::TransformationEstimationCorrentropySVD<PointXYZ, PointXYZ>);
  icp_cor->setMaximumIterations (iterations);
  icp_cor->setTransformationEstimation (trans_cor_svd);
  icp_cor->setInputSource (cloud_source_cor_svd);
  icp_cor->setInputTarget (cloud_target_cor_svd);
  
//////////////////////////////////////////////////////////////////////////////////////////////////////////
   

    int itctr=0;

 
   transformation_matrix = Eigen::Matrix4d::Identity ();
   transformation_matrix1 = Eigen::Matrix4d::Identity ();

   Eigen::Matrix4d transformation_matrix_st_svd=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_p2pl=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_p2pl_weighted=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_p2pl_lls=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_p2pl_lls_weighted=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_gicp=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_icp_nl=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_ndt=Eigen::Matrix4d::Identity ();
   Eigen::Matrix4d transformation_matrix_cor_svd=Eigen::Matrix4d::Identity ();
   

   ofstream fs3,fs4,fs5,fs6,fs7,fs8,fs9,fs10,fs11,fs_cor;
   fs3.open(path+"rmse_comparison_"+ctr+".txt",ios::app);
   fs4.open(path+"ICP_STANDARD_SVD"+"_matrix_"+ctr+".txt",ios::app);
   fs5.open(path+"ICP_P2Pl"+"_matrix_"+ctr+".txt",ios::app);
   fs6.open(path+"ICP_P2Pl_WEIGHTED"+"_matrix_"+ctr+".txt",ios::app);
   fs7.open(path+"ICP_P2Pl_LLS"+"_matrix_"+ctr+".txt",ios::app);
   fs8.open(path+"ICP_P2Pl_LLS_WEIGHTED"+"_matrix_"+ctr+".txt",ios::app);
   fs9.open(path+"GICP"+"_matrix_"+ctr+".txt",ios::app);
   fs10.open(path+"ICP_NL"+"_matrix_"+ctr+".txt",ios::app);
   fs11.open(path+"NDT"+"_matrix_"+ctr+".txt",ios::app);
   fs_cor.open(path+"coricp_matrix_"+ctr+".txt",ios::app);

  
    while(itctr<50)
    {

      // The Iterative Closest Point algorithm

      time.tic ();
      icp_st_svd->align (*cloud_source_st_svd);
      icp_p2pl->align (*cloud_source_with_normals_p2pl);
      icp_p2pl_weighted->align (*cloud_source_with_normals_p2pl_weighted);
      icp_p2pl_lls->align (*cloud_source_with_normals_p2pl_lls);
      icp_p2pl_lls_weighted->align (*cloud_source_with_normals_p2pl_lls_weighted);
      gicp->align (*cloud_source_gicp);
      icp_nl->align (*cloud_source_icp_nl);
      ndt->align (*cloud_source_ndt);
      icp_cor->align (*cloud_source_cor_svd);

      double endtime=time.toc();
      std::cout << "\nApplied " << iterations<<" iteration in " << time.toc () << " ms" << std::endl;
      std::cout<<"\n ICP(Standard SVD) fitness score = " <<icp_st_svd->getFitnessScore ();
      std::cout<<"\n ICP(P2Pl) fitness score = " <<icp_p2pl->getFitnessScore ();
      std::cout<<"\n ICP(P2Pl Weighted) fitness score = " <<icp_p2pl_weighted->getFitnessScore ();
      std::cout<<"\n ICP(P2Pl LLS) fitness score = " <<icp_p2pl_lls->getFitnessScore ();
      std::cout<<"\n ICP(P2Pl LLS Weighted) fitness score = " <<icp_p2pl_lls_weighted->getFitnessScore ();
      std::cout<<"\n ICP NON LINEAR fitness score = " <<icp_nl->getFitnessScore ();
      std::cout<<"\n NDT fitness score = " <<ndt->getFitnessScore ();
      std::cout<<"\n COrICP fitness score = " <<icp_cor->getFitnessScore ();

      fs3<<icp_st_svd->getFitnessScore()<<" "<<icp_p2pl->getFitnessScore()<<" "<<icp_p2pl_weighted->getFitnessScore()<<" "<<
      icp_p2pl_lls->getFitnessScore()<<" "<<icp_p2pl_lls_weighted->getFitnessScore()<<" "<<
      gicp->getFitnessScore()<<" "<<icp_nl->getFitnessScore ()<<" "<<ndt->getFitnessScore ()<<" "<<icp_cor->getFitnessScore()<<"\n";
      fs3.close();

      std::cout<<"\n............Computed Transformation Matrix(Standard SVD)..........."<<"ICP STANDARD SVD fitness score = " <<icp_st_svd->getFitnessScore ()<<"..............\n";
      transformation_matrix_st_svd *= icp_st_svd->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_st_svd);  // Print the transformation between original pose and current pose
      
        for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs4<<transformation_matrix_st_svd(t1,t2)<<" ";
              }
          }
        fs4<<"\n";
        fs4.close();

      std::cout<<"\n............Computed Transformation Matrix(Point to plane)..........."<<"ICP P2PL fitness score = " <<icp_p2pl->getFitnessScore ()<<"..............\n";
      transformation_matrix_p2pl *= icp_p2pl->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_p2pl);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs5<<transformation_matrix_p2pl(t1,t2)<<" ";
              }
          }
        fs5<<"\n";  
        fs5.close();

      std::cout<<"\n............Computed Transformation Matrix(Point to plane Weighted)..........."<<"ICP P2Pl Weighted fitness score = " <<icp_p2pl_weighted->getFitnessScore ()<<"..............\n";
      transformation_matrix_p2pl_weighted *= icp_p2pl_weighted->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_p2pl_weighted);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs6<<transformation_matrix_p2pl_weighted(t1,t2)<<" ";
              }
          }
        fs6<<"\n";    
        fs6.close();

      std::cout<<"\n............Computed Transformation Matrix(Point to plane LLS)..........."<<"ICP P2PL LLS fitness score = " <<icp_p2pl_lls->getFitnessScore ()<<"..............\n";
      transformation_matrix_p2pl_lls *= icp_p2pl_lls->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_p2pl_lls);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs7<<transformation_matrix_p2pl_lls(t1,t2)<<" ";
              }
          }
        fs7<<"\n";  
        fs7.close();

      std::cout<<"\n............Computed Transformation Matrix(Point to plane LLS Weighted)..........."<<"ICP P2PL LLS Weighted fitness score = " <<icp_p2pl_lls_weighted->getFitnessScore ()<<"..............\n";
      transformation_matrix_p2pl_lls_weighted *= icp_p2pl_lls_weighted->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_p2pl_lls_weighted);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs8<<transformation_matrix_p2pl_lls_weighted(t1,t2)<<" ";
              }
          }
        fs8<<"\n";  
        fs8.close();

      std::cout<<"\n............Computed Transformation Matrix(GICP)..........."<<"GICP fitness score = " <<gicp->getFitnessScore ()<<"..............\n";
      transformation_matrix_gicp *= gicp->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_gicp);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs9<<transformation_matrix_gicp(t1,t2)<<" ";
              }
          }
        fs9<<"\n";    
        fs9.close();

      std::cout<<"\n............Computed Transformation Matrix(ICP NON-LINEAR)..........."<<"ICP NL fitness score = " <<icp_nl->getFitnessScore ()<<"..............\n";
      transformation_matrix_icp_nl *= icp_nl->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_icp_nl);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs10<<transformation_matrix_icp_nl(t1,t2)<<" ";
              }
          }
        fs10<<"\n";
        fs10.close();      

      std::cout<<"\n............Computed Transformation Matrix(NDT)..........."<<"NDT fitness score = " <<ndt->getFitnessScore ()<<"..............\n";
      transformation_matrix_ndt *= ndt->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_ndt);  // Print the transformation between original pose and current pose
      
       for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs11<<transformation_matrix_ndt(t1,t2)<<" ";
              }
          }
        fs11<<"\n";  
        fs11.close();      
        
      std::cout<<"\n............Computed Transformation Matrix(CoR_ICP)..........."<<"COrICP fitness score = " <<icp_cor->getFitnessScore ()<<"..............\n";
      transformation_matrix_cor_svd *= icp_cor->getFinalTransformation ().inverse().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
      print4x4Matrix (transformation_matrix_cor_svd);  // Print the transformation between original pose and current pose
        for(int t1=0;t1<4;t1++)
          {
            for(int t2=0;t2<4;t2++)
              {
                fs_cor<<transformation_matrix_cor_svd(t1,t2)<<" ";
              }
          }
        fs_cor<<"\n";
        fs_cor.close();

         fs3.open(path+"rmse_comparison_"+ctr+".txt",ios::app);
        fs4.open(path+"ICP_STANDARD_SVD"+"_matrix_"+ctr+".txt",ios::app);
   fs5.open(path+"ICP_P2Pl"+"_matrix_"+ctr+".txt",ios::app);
   fs6.open(path+"ICP_P2Pl_WEIGHTED"+"_matrix_"+ctr+".txt",ios::app);
   fs7.open(path+"ICP_P2Pl_LLS"+"_matrix_"+ctr+".txt",ios::app);
   fs8.open(path+"ICP_P2Pl_LLS_WEIGHTED"+"_matrix_"+ctr+".txt",ios::app);
   fs9.open(path+"GICP"+"_matrix_"+ctr+".txt",ios::app);
   fs10.open(path+"ICP_NL"+"_matrix_"+ctr+".txt",ios::app);
   fs11.open(path+"NDT"+"_matrix_"+ctr+".txt",ios::app);
   fs_cor.open(path+"coricp_matrix_"+ctr+".txt",ios::app);
        

        


        itctr++;
        iterations=iterations+iter1;
   
  }
  next_iteration = false;
  fs3.close();
 
}

  return (0);
}