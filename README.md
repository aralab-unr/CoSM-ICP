# CoSM_ICP
<img src="gif_1.gif" width="1380" height="800"/>****

The last row last column in the figure above shows the CoSM ICP results. The key thing in this work is to estimate the transformation accurately under various random rotation and translation(transformation). 

Welcome to CoSM_ICP Algorithm.
Few Notes before proceeding:
1) We don't use the installed pcl library in the system. We have our own modified PCL library  (containing our correntropy Matrix implementation and the original PCL version 1.9) present in the external_libraries folder. 
2) the src/ folder contains 2 main files: CoSM_ICP_demo_Viewer.cpp and CoSM_Results_Collection.cpp.
3) CoSM_ICP_demo_Viewer.cpp is the main file which allows you to compare different methods present in the PCL library. CoSM_Results_Collection.cpp simply collects the same(transformation and RMSE's) in a file.
4) test_files contains our original implementation from scratch. We originally worked in this implementation and it's from libicp from Andreas Gieger.

Steps to Run the Program (Tested in Ubuntu 18.04).
1) Open terminal and run setup.sh in the directory. A figure window like the one shown above pops up. Hit 'space' to increase the iteration number.
2) You can run it individually as :  ./CoSM_ICP_demo_Viewer bun01.pcd 1 0.005


bun01.pcd --> filename

1 --> If you hit space in the GUI the iteration is performed '1' times. Change it to your preference.

0.005 --> voxel size to reduce the number of point clouds.

If you want to introduce outliers you can select the percentage of points that can be affected by outliers by editing line 495 in CoSM_ICP_demo_Viewer.cpp to see how well it performs in the presence of outliers. The double variable 'per' determines the percentage of data that is affected by outliers.

Please let us know if you have any questions: 

Email: ashutosh.singandhupe@gmail.com
  
  
