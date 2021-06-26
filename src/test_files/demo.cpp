

#include <iostream>
#include "icpPointToPlane.h"
#include "icpPointToPoint.h"
#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <random>
#include<math.h>
#include <Eigen/Geometry>
using namespace std;

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

int main (int argc, char** argv) {

  // define a 3 dim problem with 10000 model points
  // and 10000 template points:
for (int l=0;l<10;l++)
  {
  int32_t dim = 3;
  int32_t num = 100;
  

  double min1=-2.0,max1=2.0;
  double factor=(max1-min1)/sqrt(num);

  // allocate model and template memory
  double* M = (double*)calloc(3*num,sizeof(double));
  double* T = (double*)calloc(3*num,sizeof(double));

  // set model and template points
  cout << endl << "Creating model with 10000 points ..." << endl;
  cout << "Creating template by shifting model by (1,0.5,-1) ..." << endl;
  int32_t k=0;

  unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator2(seed2);

  std::normal_distribution<double> distribution2(0.0,3.50);

  ////Generate random for checking under various rotation and trasnlation///////
  unsigned seed_ax = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ax(seed_ax);

  std::normal_distribution<double> distribution_ax(0,3.14);

   unsigned seed_ay = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ay(seed_ay);

  std::normal_distribution<double> distribution_ay(0,3.14);

  unsigned seed_az = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_az(seed_az);

  std::normal_distribution<double> distribution_az(0,3.14);



  unsigned seed_tx = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_tx(seed_tx);

  std::normal_distribution<double> distribution_tx(0,10);

   unsigned seed_ty = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_ty(seed_ty);

  std::normal_distribution<double> distribution_ty(0,10);

  unsigned seed_tz = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator_tz(seed_tz);

  std::normal_distribution<double> distribution_tz(0,10);

  double ax1=distribution_ax(generator_ax);
  double ay1=distribution_ay(generator_ay);
  double az1=distribution_az(generator_az);
  
  double tx1=distribution_tx(generator_tx);
  double ty1=distribution_ty(generator_ty);
  double tz1=distribution_tz(generator_tz);
  
  
  Eigen::Affine3d r = create_rotation_matrix(ax1,ay1,az1);
  Eigen::Affine3d tr(Eigen::Translation3d(Eigen::Vector3d(tx1,ty1,tz1)));

  Eigen::Matrix4d m = (tr * r).matrix(); // Option 1

  m = tr.matrix(); // Option 2
  m *= r.matrix();

  Matrix AR = Matrix::eye(3);
  Matrix At(3,1);

  AR.val[0][0]=m(0,0);  AR.val[0][1]=m(0,1); AR.val[0][2]=m(0,2);
  AR.val[1][0]=m(1,0);  AR.val[1][1]=m(1,1); AR.val[1][2]=m(1,2);
  AR.val[2][0]=m(2,0);  AR.val[2][1]=m(2,1); AR.val[2][2]=m(2,2);
  
  At.val[0][0]=m(0,3);
  At.val[1][0]=m(1,3);
  At.val[2][0]=m(2,3);
  

  std::cout<<"Matrix= \n"<<m<<"\n";
  
  for (double x=min1; x<max1; x+=factor) {
    for (double y=min1; y<max1; y+=factor) {
      double z=5*x*exp(-x*x-y*y);
      M[k*3+0] = x;
      M[k*3+1] = y;
      M[k*3+2] = z;
      T[k*3+0] = x*m(0,0) + y*m(0,1)+z*m(0,2)+m(0,3);
      T[k*3+1] = x*m(1,0) + y*m(1,1)+z*m(1,2)+m(1,3);
      T[k*3+2] = x*m(2,0) + y*m(2,1)+z*m(2,2)+m(2,3);
      k++;
    }
  }

  int check = mkdir("iter_data",0777);
  
    // check if directory is created or not
    if (!check)
        printf("Directory created\n");
    else {
        printf("Unable to create directory\n");
        //exit(1);
    }
  
   // getch();
  
 //   system("dir");
  
  /////////////////////// Add outliers (at random locations with random values) /////////////////////////
  double per=0.0;
  int nop=(per*num)/100;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  std::normal_distribution<double> distribution (0,num);


  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator1(seed1);

  std::normal_distribution<double> distribution1(0,4.50);

  
 ofstream ofs1,ofs2;

  ofs2.open("iter_data/idx_att.txt",ios::app);
  for(int i=0;i<nop;i++)
  {
     int v1=rand()%(3*num);
     ofs2<<v1/3<<"\n";
     double v2=distribution1(generator1);
     T[v1]=T[v1]+v2;
  }
  ofs2.close();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

 /////////////Compute variance of data (Model and template) //////////////////////////////////////////
  

 ////////////////////////////////////////////////////////////////////////////////////////////////////

  
  ofs1.open("iter_data/original_data.txt",ios::app);
  for(int i=0;i<num;i++)
  {
  	for(int j=0;j<dim;j++)
  	{
  	  ofs1<<M[i*3+j]<<" ";
  	}

  	for(int j=0;j<dim;j++)
  	{
  	  ofs1<<T[i*3+j]<<" ";
  	}
  	ofs1<<"\n";
  }
  ofs1.close();

 
  // start with identity as initial transformation
  // in practice you might want to use some kind of prediction here
  Matrix R = Matrix::eye(3);
  Matrix t(3,1);

  Matrix plR = Matrix::eye(3);
  Matrix plt(3,1);

  // run point-to-plane ICP (-1 = no outlier threshold)
  cout << endl << "Running ICP (point-to-plane, no outliers)" << endl;
  //IcpPointToPoint icp(M,num,dim);
  IcpPointToPlane icp_pl(M,num,dim);
 // icp.corr_trigger_rms=1.0;
  
 // double residual = icp.fit(T,num,R,t,AR,At,-1);
  
  double residual2 = icp_pl.fit(T,num,plR,plt,AR,At,-1);

  //results
  cout << endl << "Transformation results(p2pl):" << endl;
  cout << "R:" << endl << plR << endl << endl;
  cout << "t:" << endl << plt << endl << endl;
  cout << "Residual:"<<residual2;

  // free memory
  free(M);
  free(T);
}
  // success
  return 0;
}

