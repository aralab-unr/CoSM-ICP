
#ifndef ICP_POINT_TO_PLANE_H
#define ICP_POINT_TO_PLANE_H

#include "icp.h"
#include <Eigen/Geometry>

class IcpPointToPlane : public Icp {

public:
  std::ofstream ofs;
  int iter_ctr=0;
  IcpPointToPlane (double *M,const int32_t M_num,const int32_t dim,const int32_t num_neighbors=10,const double flatness=5.0) : Icp(M,M_num,dim) {
    M_normal = computeNormals(num_neighbors,flatness);
  }

  virtual ~IcpPointToPlane () {
    free(M_normal);
  }

private:

	double fitStep (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active,double delta);
	double fitStepCorr (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active,double delta);
  
	std::vector<int32_t> getInliers (double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const double indist);
	double getResidual(double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const std::vector<int> &active);
	// utility functions to compute normals from the model tree
	void computeNormal (const kdtree::KDTreeResultVector &neighbors,double *M_normal,const double flatness);
	double* computeNormals (const int32_t num_neighbors,const double flatness);

	// normals of model points
	double *M_normal;
};

#endif // ICP_POINT_TO_PLANE_H
