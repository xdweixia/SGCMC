% Generate Balanced two-moon data
clc
clear all
addpath([pwd, '/funs']);
N1=500; N2=500;

%% X1: Generated Raw representation; Y: ground_truth
[X1,Y] = twomoon_gen(N1, N2);

%% A: Constructed Graph Structure
sigma=optSigma(X1);
options.KernelType = 'Gaussian';
options.t = 0.5;
A = constructKernel(X1,X1,options);

%% Construct The Euler Representation for X1
alpha = 1.1;
Data = X1';
[E_Data] = Euler_transform_1D(Data,alpha); % Input: d*N
X2 = E_Data'; % Euler representation, N*d
