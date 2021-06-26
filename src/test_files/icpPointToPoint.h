
#ifndef ICP_POINT_TO_POINT_H
#define ICP_POINT_TO_POINT_H

#include "icp.h"
#include <fstream>
#include <Eigen/Dense>

class IcpPointToPoint : public Icp 
{

public:
  std::ofstream ofs;
  int iter_ctr=0;
  IcpPointToPoint (double *M,const int32_t M_num,const int32_t dim) : Icp(M,M_num,dim) {}
  virtual ~IcpPointToPoint () {}
  
private:

  double fitStep (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active,double delta);
  double fitStepCorr (double *T,const int32_t T_num,Matrix &R,Matrix &t,const std::vector<int32_t> &active,double delta);
  std::vector<int32_t> getInliers (double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const double indist);
  double getResidual(double *T,const int32_t T_num,const Matrix &R,const Matrix &t,const std::vector<int> &active);
};

#endif // ICP_POINT_TO_POINT_H
