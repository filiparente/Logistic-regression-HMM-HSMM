function [model, lkh, iter] = hmm_train(params, x, X, max_iterations, tolerance, mode, Vk)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% Input:
%   x: 1 x n integer vector which is the sequence of observations
%   params: model or lambdas vector
%   mode - only applicable if the logistic regression is being used. 
%   if true, all features are used to estimate the weights of the
%   logistic regression; if false, only the features of the timestamps
%   associated with a transition, according to the state sequence
%   estimated by the EM algorithm.
% Output:s
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).

%default values: tolerance = 1e-20; max_iterations = 100

k = length(params.lambdas);
size_A = size(params.A);
if size_A(end)>k
    [model, lkh, iter] = logit_poissonhmmEm(params, x, X, max_iterations, tolerance, mode, Vk);
else
    [model, lkh, iter] = poissonhmmEm(params, x, max_iterations, tolerance, Vk);
end
    

end
        