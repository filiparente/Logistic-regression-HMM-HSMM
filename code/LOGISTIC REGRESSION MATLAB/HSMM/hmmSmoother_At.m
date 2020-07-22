function [At_est, trans_pos, gamma, alpha, beta, c] = hmmSmoother_At(M, At, s)
% Implmentation function HMM smoothing alogrithm.
% Unlike the method described in the book of PRML, the alpha returned is the normalized version: gamma(t)=p(z_t|x_{1:T})
% Computing unnormalized version gamma(t)=p(z_t,x_{1:T}) is numerical unstable, which grows exponential fast to infinity.
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 start prior probability
% Output:
%   gamma: k x n matrix of posterior gamma(t)=p(z_t,x_{1:T})
%   alpha: k x n matrix of posterior alpha(t)=p(z_t|x_{1:T})
%   beta: k x n matrix of posterior beta(t)=gamma(t)/alpha(t)
%   c: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
[K,T] = size(M);
%At = A';
%At_ = permute(At, [K K-1 K+1]);
At_ = permute(At, [2 1 3]);
c = zeros(1,T); % normalization constant
alpha = zeros(K,T);
[alpha(:,1),c(1)] = normalize(s.*M(:,1),1);
for t = 2:T
    %alpha(:,t) = At_(:,:,t-1)*alpha(:,t-1).*M(:,t);
    [alpha(:,t),c(t)] =  normalize((At_(:,:,t-1)*alpha(:,t-1)).*M(:,t),1);  % 13.59
end
beta = ones(K,T);
for t = T-1:-1:1
    beta(:,t) = At(:,:,t)*(beta(:,t+1).*M(:,t+1))/c(t+1);   % 13.62
end
%state posteriors
gamma = alpha.*beta;                  % 13.64
%sum(gamma,1)
%gamma = normalize(gamma,1);

%calculate likelihood of observations
likelihood = sum(alpha(:,end));

%transition posteriors
trans_pos = zeros(K,K,T);
At_est = zeros(K,K,T);
for t=1:T-1
    for i=1:K
        for j=1:K
            trans_pos(i,j,t) = alpha(i,t)*c(t)*At(i,j,t)*M(j,t+1)*beta(j,t+1)*c(t+1);
            %trans_pos(i,j,t) = alpha(i,t)*At(i,j,t)*M(j,t+1)*beta(j,t+1); 
        end
    end
    sum_ = sum(sum(trans_pos(:,:,t)));
    
    trans_pos(:,:,t) = trans_pos(:,:,t)./sum_;
    gamma(gamma<1e-200) = gamma(gamma<1e-200)+1e-100;
    At_est(:,:,t) = trans_pos(:,:,t)./(repmat(gamma(:,t),1,K));

    if size(At_est(sum(At_est(:,:,t),2)<1e-1,:,t),1)~=0
        At_est(sum(At_est(:,:,t),2)<1e-100,:,t) = repmat(ones(1,K)./K, length(At_est(sum(At_est(:,:,t),2)<1e-100)),1);
    end
end

