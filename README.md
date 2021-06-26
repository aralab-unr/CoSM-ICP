# CoSM_ICP
<img src="gif_1.gif" width="1380" height="800"/>****

The last row last column in the figure above shows the CoSM ICP results.

Welcome to CoSM_ICP Algorithm.
Few Notes before proceeding:
1) We don't use the installed pcl library in the system. We have our own modified PCL library (containing our correntropy Matrix implementation) present in the external_libraries folder. 
2) the src/ folder contains 2 main files: CoSM_ICP_demo_Viewer.cpp and CoSM_Results_Collection.cpp.
3) CoSM_ICP_demo_Viewer.cpp is the main file which allows you to compare different methods present in the PCL library. CoSM_Results_Collection.cpp simply collects the same(transformation and RMSE's) in a file.
4) test_files contains our original implementation from scratch. We originally worked in this implementation and it's from libicp from Andreas Gieger.

Steps to Run the Program.
1) Open terminal and run setup.sh in the directory. A figure window like the one shown above pops up. Hit 'space' to increase the iteration number.
2) You can run it individually as :  ./CoSM_ICP_demo_Viewer bun01.pcd 1 0.005


bun01.pcd --> filename

1 --> iteration increase counter when you hit space.

0.005 --> voxel size to reduce the nmber of point clouds.
  
  
