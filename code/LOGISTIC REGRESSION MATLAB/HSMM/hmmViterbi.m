function [z,p] = hmmViterbi(x, model)
% Implmentation function of Viterbi algorithm. 
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 starting probability (prior)
% Output:
%   z: 1 x n latent state
%   p: 1 x n probability
% Written by Mo Chen (sth4nth@gmail.com).
A = model.At;
lambdas = model.lambdas;
k = length(lambdas);
s = model.s;
n = size(x,2);

M = zeros(k,n);
for i=1:k
    M(i,:) = poisspdf(x,lambdas(i));
end 
M = normalize(M,1);

Z = zeros(k,n);
A = log(A);
M = log(M);
Z(:,1) = 1:k;
v = log(s(:))+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A,v),[],1);    % 15.68
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[v,idx] = max(v);
z = Z(idx,:);
p = exp(v);


%% OLD CODE
% function z = hmmViterbi(x, model)
% % Viterbi algorithm calculated in log scale to improve numerical stability.
% % This is a wrapper function which transform input and call underlying algorithm
% % Input:
% %   x: 1 x n integer vector which is the sequence of observations
% %   model:  model structure
% % Output:
% %   z: 1 x n latent state
% % Written by Mo Chen (sth4nth@gmail.com).
% At = model.At;
% lambdas = model.lambdas;
% k=length(lambdas);
% s = model.s;
% n = size(x,2);
% %d = max(x);
% %X = sparse(x,1:n,1,d,n);
% M = zeros(k,n);
% for i=1:k
%     M(i,:) = poisspdf(x,lambdas(i));
% end 
% M = normalize(M,1);
% %M = E*X;
% z = hmmViterbi_(M, At, s);
%end