function [x, st, model] = poissonhmmRnd(k, n, lambdas)
% Generate a data sequence from a hidden Markov model.
% Input:
%   d: dimension of data
%   k: dimension of latent variable
%   n: number of data
% Output:
%   X: d x n data matrix
%   model: model structure
% Written by Mo Chen (sth4nth@gmail.com).

A = normalize(rand(k,k),2);
%E = normalize(rand(k,d),2);
s = normalize(rand(k,1),1);
x = zeros(1,n);
st = zeros(1,n);
z = discreteRnd(s);
%x(1) = discreteRnd(E(z,:));
st(1) = z;
x(1) = poissrnd(lambdas(z));

for i = 2:n
    z = discreteRnd(A(z,:));
    st(i) = z;
    x(i) = poissrnd(lambdas(z)); 
end
model.A = A;
model.lambdas = lambdas;
model.s = s;