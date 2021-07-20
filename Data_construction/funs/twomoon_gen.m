function [input,y] = twomoon_gen(num1, num2, sigma_noise, horizonal, vertical)

if nargin == 1
    num2 = num1;
end;
if nargin <= 2
    sigma_noise = 0.12;
end;
if nargin <= 3
    level = 0.35;
    upright = 0.15;
else
    level = 0.32+horizonal;
    upright = 0.15+vertical;
end;
t=pi:-pi/(num1-1):0;
input(1:num1, 1) = cos(t)'+randn(num1,1)*sigma_noise - level;
input(1:num1, 2) = sin(t)'+randn(num1,1)*sigma_noise - upright;

t=pi:pi/(num2-1):2*pi;
input(num1+1:num1+num2, 1) = cos(t)'+randn(num2,1)*sigma_noise + level;
input(num1+1:num1+num2, 2) = sin(t)'+randn(num2,1)*sigma_noise + upright;

y = [ones(num1,1); 2*ones(num2,1)];

 

