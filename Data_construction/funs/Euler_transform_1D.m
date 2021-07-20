function [E_Data] = Euler_transform_1D(Data,alpha)
% 1D
% Data: ndim*Nsamlpes 
% ndim : data dimension, Nsamlpes : the number of data
t_min = min(Data, [], 1); t_max = max(Data, [], 1);
norm_Data = (Data - repmat(t_min,[size(Data,1) 1]))./(repmat(t_max - t_min, [size(Data,1) 1]));
E_Data = inv(sqrt(2))*exp(1i*alpha*pi*norm_Data);
end
