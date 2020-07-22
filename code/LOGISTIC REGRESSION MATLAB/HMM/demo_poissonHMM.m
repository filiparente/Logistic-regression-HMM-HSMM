% demos for HMM 
d = 3;
k = 2;
n = 10000;
[x, model] = hmmRnd(d, k, n);
%%    % Run viterbi algorithm
% first input is the model structure which contains
% model.s: k x 1 start probability vector
% model.A: k x k transition matrix
% model.E: k x d emission matrix
% second input is an 1 x n integer vector which is the sequence of observations
% first output is an 1 x n discrete vector with the decoded latent
% state sequence
% second output is the loglikelihood

%model = struct;
%model.s = estimated_pi;
%model.A = estimated_T;
%model.E
[z, llh] = hmmViterbi(model, x);
%z = hmmViterbi(x,model);
%%
[alpha,llh] = hmmFilter(x,model);
%%
[gamma,alpha,beta,c] = hmmSmoother(x,model);
%%
[model, llh] = hmmEm(x,k);
plot(llh)