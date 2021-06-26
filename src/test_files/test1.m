clc;
clear all;
close all;

figure(1);

for i=1:200
s1='rmse_comaprison_';
rmsee1=load('test_results/bunny_dataset/ICP_STANDARD_SVD/rmse_comparison_'+string(i)+'.txt');
plot(rmsee1(2:1000,1),'-o');
hold on;
plot(rmsee1(2:1000,2),'-x');
clf;

end