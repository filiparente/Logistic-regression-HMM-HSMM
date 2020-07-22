function models = logit_poissonHsmmEm(x, X, init, states, K, d, dim, N, or_transf_A, or_lambdas, compare)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% Input:
%   X: n_features x n, feature vector
%   x: 1 x n integer vector which is the sequence of observations
%   init: model or n_states
% Output:s
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
n = size(x,1);
%d = max(x);
%X = sparse(x,1:n,1,d,n);
if isstruct(init)   % init with a model
    A = init.A;
    lambdas = init.lambdas;
    s = init.s;
    % TODO
elseif numel(init) == 1  % random init with latent k
    k = init;
    [At,B,P,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,k,d,K,dim*ones(N,1));
end
%matriz de transiçao inicial toda igual, ou random? tentei toda igual mas
%os resultados nao foram muito bons, a matriz estimada tambem era quase
%sempre a mesma, vou agora tentar random.
%At = repmat(A, 1,1,n); %all transition matrix for all t are equal
    
models = {};
        
[model, lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest, store_GAMMA, store_ALPHA, lkh, ll, ir]=hsmm_new_(lambdas, PAI,At,B,P,x,100,dim*ones(N,1), 1e-20, Vk, X, or_transf_A, or_lambdas);
model.store_GAMMA = store_GAMMA;
model.store_ALPHA = store_ALPHA;
models{1} = model;

if compare
    [lambdas_est2, lambdas_tt2, PAI_est2,A_est2,B_est2,P_est2,Qest2, store_GAMMA2, store_ALPHA2, lkh2, ll2, ir2]=hsmm_new(lambdas, PAI,At(:,:,1),B,P,x,100,dim*ones(N,1), 1e-10, Vk);

    model2.A = A_est2;
    model2.lambdas = lambdas_est2;
    model2.s = PAI_est2;
    model2.P = P_est2;
    model2.Qest = Qest2;
    model2.B = B_est2;
    model2.store_GAMMA = store_GAMMA2;
    model2.store_ALPHA = store_ALPHA2;
    
    models{2} = model2;
end

end
