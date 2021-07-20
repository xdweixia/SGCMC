% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function W = constructW_PKN(X, k, issymmetric)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 200;
end

[dim, n] = size(X);
D = L2_distance_1(X, X);
[dumb, idx] = sort(D, 2); % sort each row

W = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

if issymmetric == 1
    W = (W+W')/2;
end




% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end





