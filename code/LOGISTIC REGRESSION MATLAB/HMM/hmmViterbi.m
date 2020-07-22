function z = hmmViterbi(x, model)
% Viterbi algorithm calculated in log scale to improve numerical stability.
% This is a wrapper function which transform input and call underlying algorithm
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   model:  model structure
% Output:
%   z: 1 x n latent state
% Written by Mo Chen (sth4nth@gmail.com).
At = model.At;
lambdas = model.lambdas;
k=length(lambdas);
s = model.s;
n = size(x,2);
%d = max(x);
%X = sparse(x,1:n,1,d,n);
M = zeros(k,n);
for i=1:k
    M(i,:) = poisspdf(x,lambdas(i));
end 
M = normalize(M,1);
%M = E*X;
z = hmmViterbi_(M, At, s);