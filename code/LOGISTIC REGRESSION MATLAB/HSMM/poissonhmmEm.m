function [model, lkh, iter] = poissonhmmEm(params, x, max_iterations, tolerance, Vk)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   init: model or lambdas vector
% Output:s
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
dim = size(x,2);
N = size(x,1);

%X = sparse(x,1:n,1,d,n);

if isstruct(params)   % params with a model
    A = params.A;
    lambdas = params.lambdas;
    PAI = params.PAI;
    B = params.B;
    k = size(A,1);
else%if numel(params) == 1  % random params with latent k
    %k = params;
    %from lambdas vector retrieve k
    k = numel(params);
    A = normalize(rand(k,k),2);
    lambdas = params;
    PAI = normalize(rand(k,1),1);
end
%M = E*X;
M = zeros(k,dim);
for i=1:k
    M(i,:) = poisspdf(x,lambdas(i));
    M(i,:) = M(i,:)/(sum(poisspdf(Vk, lambdas(i))));
end
M = M + 1e-100; %to avoid numerical errors (Nan's in gamma)
llh = -inf(1,max_iterations);
lambdas_tt = {};

for iter = 2:max_iterations
%     E-step
    [gamma,alpha,beta,c] = hmmSmoother_(M,A,PAI);

    llh(iter) = sum(log(c(c>0)));
    if llh(iter)-llh(iter-1) < tolerance*abs(llh(iter-1)); break; end   % check likelihood for convergence
%     M-step 
    A = normalize((A.*(alpha(:,1:dim-1)*bsxfun(@times,beta(:,2:dim).*M(:,2:dim),1./c(2:end))')+1e-100),2);      % 13.19
    s = gamma(:,1);                                                                             % 13.18
    %M = bsxfun(@times,gamma*X',1./sum(gamma,2))*X;
    lambdas = sum(M*x',2)./sum(M,2);
    lambdas_tt{iter} = lambdas;
    for i=1:k
        M(i,:) = poisspdf(x,lambdas(i));
        M(i,:) = M(i,:)/(sum(poisspdf(Vk, lambdas(i))));
    end 
    M = M + 1e-100; %to avoid numerical errors (Nan's in gamma)
end
lkh = llh(2:iter);

B = zeros(k, length(Vk));
for i=1:k
 %ss(i)=sum(log(1:i));
 B(i,:) = poisspdf(Vk, lambdas(i))';
 B(i,:)=B(i,:)/sum(B(i,:));
end

model.A = A;
model.s = s';
model.lambdas = lambdas';
model.lambdas_tt = lambdas_tt;
model.B = B;

aux_model.At=A;
aux_model.lambdas= lambdas;
aux_model.s = s;

[Qest, ~] = hmmViterbi(x, aux_model);
model.Qest = Qest';
model.store_GAMMA = gamma;

iter=iter-1;
