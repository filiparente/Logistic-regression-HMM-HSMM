clear all;
rng(1); %fix seed for reproducibility

n_features = 2;
n_states = 2;
T = 1000;

lambdas = [8, 22];

% GENERATE FEATURE DATA X
X = zeros(n_features, T);

% INPUTS?
% Parameters of the normal distributions (mean+std), one for each state/class
mu = [6 0;-2 0;10 3; -14 -5];
sigma = [0.6 0.01]; % shared diagonal covariance matrix

% Set up a Markov model to simulate the transitions between the feature
% vector X, for each class
transition_matrix = [0.985 0.005 0.005 0.005; 0.005 0.985 0.005 0.005; 0.005 0.005 0.985 0.005; 0.005 0.005 0.005 0.985];

%we don't care about the emissions, only the transitions
emission_matrix = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;
                    7/12, 1/12, 1/12, 1/12, 1/12, 1/12;
                    7/12, 1/12, 1/12, 1/12, 1/12, 1/12;
                    7/12, 1/12, 1/12, 1/12, 1/12, 1/12
                   ];

[~,states] = hmmgenerate(T, transition_matrix, emission_matrix); %no pi, it always starts at state 1

% Using the state sequence for X, sample from a normal distribution
% with mean=5 when in state 1, and from another normal distribution
% with mean=-5 when in state 2.
X(:, states==1) = mvnrnd(mu(1,:),sigma,length(X(states==1)))';
X(:, states==2) = mvnrnd(mu(2,:),sigma,length(X(states==2)))';
X(:, states==3) = mvnrnd(mu(3,:),sigma,length(X(states==3)))';
X(:, states==4) = mvnrnd(mu(4,:),sigma,length(X(states==4)))';

X = X';
% Plot features
figure
scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
grid
X = X';

%betas = [2 1; 3 -3; 0 0];
%betas = [0.3 1 -0.2 -6; 0.5 3 -0.5 2; 0 0 0 0]; %R^(p+1)*k^2 , p=2, k=2, (3x4)
%betas = [-1 -0.5 0.7 0.9; -0.6 0.01 0.01 0.3; 0 0 0 0];
betas = [0.8 -0.2 1 1; -2 1 2 -2; 0 0 0 0];

%betas = [1 -2; 1 -3; 0 0];
%get an orthogonal vector of B11
%o_beta11 = null(betas(:,1).');
%beta1 = [betas(:,1), o_beta11(:,1)];

%get an orthogonal vector of B11
%o_beta21 = null(betas(:,2).');
%beta2 = [betas(:,2), o_beta21(:,1)];

%betas=[beta1, beta2];
%betas = [betas(:,1:2), orth(betas(:,1:2))];%o_beta11(:,1), o_beta21(:,1)];


% Generate the true state sequence from a predefined (random) HMM
% transition matrix multiplied by a logistic regression factor, 
% using the true betas(defined above) and the feature vector X
pi = generate_random_initial_dist(n_states);
%A = generate_random_transition_matrix(n_states); %uniform: ones(n_states)/n_states;

%   betas = normrnd(mu, sigma,n_features+1,n_states);
%   beta = normrnd(mu, sigma,n_features+1,1);
%   betas = repmat(beta, 1, n_states);
%   mu = [5 0;-5 0];
%   sigma = [0.5 0.01]; % shared diagonal covariance matrix
%   gm = gmdistribution(mu,sigma);
%   betas = [0 0; 2 -2; 0 0]; 
%   [X,compIdx] = random(gm,T);

% Sample first state from pi
state = sampleDiscrete(pi);
state_seq = zeros(T,1);
obs_seq = zeros(T,1);

p = zeros(n_states, 1);
store_p = zeros(T, n_states);

%f = ones(1,n_features-1)/(n_features-1); %one hot enconding feature
%vector (jogo do sporting, benfica, etc)
or_transf_A = zeros(n_states, n_states, T);
for t=1:T
    %selected_feature = sampleDiscrete(f); %one hot encoding feature
    %vector
    %X(:,i) = normrnd(mu,sigma,n_features,1);
    %disp(selected_feature);

%         for j=1:n_states
%             sum_ = 0;
%             for k=1:n_states
%                 sum_ = sum_ + logit(X(:,t),k, betas);
%             end
%             %multiplication between the HMM transition matrix and the
%             %logistic regression factor (probability of next states)
%             p(j) = A(state, j)*(logit(X(:,t),j, betas)/sum_);
%         end
%         
%         %normalize
%         p = p/sum(p);
    %i = state;
    for i=1:n_states
        for j=1:n_states
            column = get_columns(n_states,i,j);
            columns = get_columns(n_states, i);
            or_transf_A(i,j,t) = exp(betas(:,column)'*[X(:,t);1])/sum(exp(betas(:,columns)'*[X(:,t);1]));
        end    
    end
    %or_transf_A(:,:,t) = A.*or_transf_A(:,:,t)';
    %or_transf_A(:,:,t) = normalize(or_transf_A(:,:,t),2);

    p = or_transf_A(state,:,t);
    
    if t==1
        store_p(t,:)=pi;
    end
    if t~=T 
        store_p(t+1,:) = p;
    end
    state_seq(t) = state;
    obs_seq(t) = poissrnd(lambdas(state));

    state = sampleDiscrete(p');    
end

% Test with a state_seq which is a shifted version of the states that
% generated X
 states=states';
 %delay = 20; %20 samples
 %state_seq = delayseq(states,delay);
 %state_seq(state_seq==0) = states(1);
%%
[model, llh] = logit_poissonhmmEm(obs_seq', X, n_states, states);
est_state_seq = hmmViterbi(obs_seq', model);
delay = 0;

figure
subplot(3,1,1);
plot(states, 'g')
%hold on
%plot(state_seq)
%hold on
%lgd = legend('True state sequence X', 'True state sequence HMM');
lgd = legend('True state sequence X');
lgd.Location = 'northeast';
ylim([0.9 2.3])
ylabel('State')
title(['Delay = ', num2str(delay), ' samples'])
subplot(3,1,2);
plot(state_seq, 'b')
lgd = legend('True state sequence HMM');
lgd.Location = 'northeast';
ylim([0.9 2.3])
ylabel('State')
subplot(3,1,3); 
plot(est_state_seq, 'y')
ylim([0.9 2.3])
lgd = legend('Estimated state sequence HMM');
lgd.Location = 'northeast';
ylabel('State')
xlabel('t')

%mean_matrices(mat2cell(model.At, n_states,n_states, ones(T,1)))

%MSE MATRIZ DE TRANSIÇÃO
disp('MSE MATRIZ DE TRANSIÇÃO');
tmp = (or_transf_A - model.At).^2;
sum(tmp(:))/numel(or_transf_A)

%MSE LOG ODDS REAL BETAS
real_log_odds = [];
k=n_states;
for s=1:k
        logit_model.W = betas(:,s*k-1:s*k);
        [~, log_odds] = logitMnPred(logit_model, X);
        real_log_odds = [real_log_odds; log_odds];
end
%logit_model.W = betas;
%[~, log_odds] = logitMnPred(logit_model, X); %y is the predicted state sequence and P is the state posteriors according to the logistic regression model
    
disp('MSE LOG ODDS');
mse(real_log_odds, model.log_odds)

% Plot features with the original and estimated decision boundaries
X=X';

figure
%scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
% Plot feature points from the first class
scatter(X(states == 1,1), X(states == 1,2), 50, 'b', '.')
% Plot feature points from the second class.
hold on;
scatter(X(states == 2,1), X(states == 2,2), 50, 'r', '.')
% Plot feature points from the first class
hold on;
scatter(X(states == 3,1), X(states == 3,2), 50, 'g', '.')
% Plot feature points from the second class.
hold on;
scatter(X(states == 4,1), X(states == 4,2), 50, 'c', '.')

betas = [betas(end,:);betas];
betas = betas(1:end-1,:);
%plot true decision boundary
hold on
plot_decision_boundary(betas(:,1:2));
hold on
plot_decision_boundary(betas(:,3:4));
hold on
%plot estimated decision boundary
plot_decision_boundary(model.betas(:,1:2));
hold on
%plot estimated decision boundary
plot_decision_boundary(model.betas(:,3:4));
hold on
legend('Feature points from first class (11)', 'Feature points from second class (12)', 'Feature points from third class (21)', 'Feature points from forth class (22)', 'True decision boundary 11=12', 'True decision boundary 21=22', 'Estimated decision boundary 11=12', 'Estimated decision boundary 21=22')
title(['Delay = ', num2str(delay), ' samples'])
