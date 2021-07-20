function [data label]= threering(N1, N2, N3, degrees, start, ~)
% Generate "two spirals" dataset with N instances.
% degrees controls the length of the spirals
% start determines how far from the origin the spirals start, in degrees
% noise displaces the instances from the spiral. 
%  0 is no noise, at 1 the spirals will start overlapping

    if nargin < 4
        degrees = 570;
    end
    if nargin < 5
        start = 90;
    end
    if nargin < 6
        noise = 0.25;
    end  
    
    deg2rad = (2*pi)/360;
    start = start * deg2rad;

    
    n = start + sqrt(rand(N1,1)) * degrees * deg2rad;   
    d1 = [(-cos(n))+rand(N1,1)*noise (sin(n))+rand(N1,1)*noise];
    
    n = start + sqrt(rand(N2,1)) * degrees * deg2rad;
    d2 = [cos(n).*2+rand(N2,1)*noise -sin(n).*2+rand(N2,1)*noise];
    
    n = start + sqrt(rand(N3,1)) * degrees * deg2rad;      
    d3 = [cos(n).*3+rand(N3,1).*noise  -sin(n).*3+rand(N3,1)*noise];
    
    data = [d1;d2;d3];l1=ones(N1,1);l2=2*ones(N2,1);l3=3*ones(N3,1); label=[l1;l2;l3];
%     scatter(data(:,1), data(:,2), 20, data(:,3)); axis square;
    
end