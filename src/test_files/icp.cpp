

#include "icp.h"

using namespace std;

Icp::Icp (double *M,const int32_t M_num,const int32_t dim) 
:m_dim(dim), m_max_iter(200), m_min_delta(1e-4) 
{
   // gnum_ctr=0;
	// check for correct dimensionality
	if (dim!=2 && dim!=3)
	 {
		cout << "ERROR: LIBICP works only for data of dimensionality 2 or 3" << endl;
		m_kd_tree = 0;
		return;
	}
	// check for minimum number of points
	if (M_num<5) {
		cout << "ERROR: LIBICP works only with at least 5 model points" << endl;
		m_kd_tree = 0;
		return;
	}

	// copy model points to M_data
	m_kd_data.resize(boost::extents[M_num][dim]);
	for (int32_t m=0; m<M_num; m++)
		for (int32_t n=0; n<dim; n++)
		  m_kd_data[m][n] = (float)M[m*dim+n];

	// build a kd tree from the model point cloud
	m_kd_tree = new kdtree::KDTree(m_kd_data);
}

Icp::~Icp () 
{
  if (m_kd_tree)
    delete m_kd_tree;
}

double Icp::fit( double *T,const int32_t T_num,Matrix &R,Matrix &t,Matrix &aR,Matrix &at ,double indist)
{
	// make sure we have a model tree
	if (!m_kd_tree) {
		cout << "ERROR: No model available." << endl;
		return 0;
	}

	// check for minimum number of points
	if (T_num<5) {
		cout << "ERROR: Icp works only with at least 5 template points" << endl;
		return 0;
	}

	// set active points
	vector<int32_t> active;
	if (indist<=0) 
	{
		active.clear();
		for (int32_t i=0; i<T_num; i++)
	  	active.push_back(i);
	} 
	else 
	{
	active = getInliers(T,T_num,R,t,indist);
	}

	// run icp
	fitIterate(T,T_num,R,t,aR,at,indist);

	return getResidual(T,T_num,R,t,m_active);
}

void Icp::fitIterate( double *T,const int32_t T_num,Matrix &R,Matrix &t,Matrix &aR,Matrix &at,double indist)
{
	ofstream ofs3,ofs4;
	gnum_ctr=0;
	if(indist<=0)
	{
		m_active.clear();
		m_active.resize(T_num);
		for(int32_t i=0;i<T_num;i++)
		{
			m_active[i] = i;
		}
		m_inlier_ratio = 1;
	}
	double delta = 1000;
	int32_t iter;
    double rmse_at;
    ofs3.open("iter_data/rmse_normal_icp.txt",ios::app);
	std::cout<<"\nMaxiteration="<<m_max_iter;
	for(iter=0; iter<m_max_iter ; iter++)
	{
		
		//std::cout<<"\nindist="<<indist;
		flagsc=new int[T_num];
		sum_corr_val=0;
		for(int i=0;i<m_active.size();i++)flagsc[i]=0;
		if(indist>0)
		{
			indist = std::max(indist*0.9,0.05);
			m_active = getInliers(T,T_num,R,t,indist);
			m_inlier_ratio = (double)m_active.size()/T_num;
		}
        
        curr_rmse=getResidual(T,T_num,R,t,m_active);
		delta=fitStep(T,T_num,R,t,m_active,delta);
		rmse_at=getResidual(T,T_num,R,t,m_active);

		std::cout<<"\ntrans_error= "<<(at-t).l2norm()<<" rotation error="<<(aR-R).l2norm();
		std::cout<<"\n rmse_at="<<rmse_at<<" ";
		ofs3<<rmse_at<<" ";
		delete flagsc;

	}
	ofs3<<"\n";
	ofs3.close();
    
    // results
  cout << endl << "Transformation results for standard ICP:" << endl;
  cout << "R:" << endl << R << endl << endl;
  cout << "t:" << endl << t << endl << endl;
  cout << "Residual:"<<rmse_at;

   std::cout<<"\n \n \n";
  R = Matrix::eye(3);
  t.zero();
   gnum_ctr=0;
	if(indist<=0)
	{
		m_active.clear();
		m_active.resize(T_num);
		for(int32_t i=0;i<T_num;i++)
		{
			m_active[i] = i;
		}
		m_inlier_ratio = 1;
	}
	//delta = 1000;
	//int32_t iter;
     //double rmse_at;
	 ofs4.open("iter_data/rmse_corr_icp.txt",ios::app);
	for(iter=0; iter<m_max_iter; iter++)
	{
		//std::cout<<"\nindist="<<indist;
		flagsc=new int[T_num];
		sum_corr_val=0;
		for(int i=0;i<m_active.size();i++)flagsc[i]=0;
		if(indist>0)
		{
			indist = std::max(indist*0.9,0.05);
			m_active = getInliers(T,T_num,R,t,indist);
			m_inlier_ratio = (double)m_active.size()/T_num;
		}
        
        //curr_rmse=getResidual(T,T_num,R,t,m_active);
		delta=fitStepCorr(T,T_num,R,t,m_active,delta);
		rmse_at=getResidual(T,T_num,R,t,m_active);
		std::cout<<"\ntrans_error= "<<sqrt(((at-t)*~(at-t)).val[0][0])<<" rotation error="<<sqrt((aR-R).l2norm());
		std::cout<<"\n corrrmse_at="<<rmse_at<<" ";
		ofs4<<rmse_at<<" ";
		delete flagsc;

	}
	ofs4<<"\n";
	ofs4.close();

	// results
  cout << endl << "Transformation results for COrrICP:" << endl;
  cout << "R:" << endl << R << endl << endl;
  cout << "t:" << endl << t << endl << endl;
  cout << "Residual_Corr:"<<rmse_at<<" ";
}
