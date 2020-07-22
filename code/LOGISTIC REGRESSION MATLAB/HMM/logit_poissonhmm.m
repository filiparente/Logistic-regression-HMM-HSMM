function [est_A, or_transf_A, state_seq, states, X, betas, A, pi] = logit_poissonhmm(n_states, n_features, T)
    %% SAMPLING: Define betas and the original setup A and pi. Obtain X and state_seq
    
    % INPUTS   
    %n_states = 2;
    %n_features = 2;
    %T = 1000;  
    rng('default'); %fix seed for reproducibility
    
    % GENERATE FEATURE DATA X
    clear p;
   
    X = zeros(n_features, T);
    
    lambdas = [8,22];

    % INPUTS?
    % Parameters of the normal distributions (mean+std), one for each state/class
    mu = [2 0;-2 0];
    sigma = [0.6 0.01]; % shared diagonal covariance matrix
    
    % Set up a Markov model to simulate the transitions between the feature
    % vector X, for each class
    transition_matrix = [0.99 0.01; 0.01 0.99];
    betas = [0 0; -3 3; 0 0];

    %we don't care about the emissions, only the transitions
	emission_matrix = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;
                        7/12, 1/12, 1/12, 1/12, 1/12, 1/12];
    
    [~,states] = hmmgenerate(T, transition_matrix, emission_matrix); %no pi, it always starts at state 1
    
    % Using the state sequence for X, sample from a normal distribution
    % with mean=5 when in state 1, and from another normal distribution
    % with mean=-5 when in state 2.
    X(:, states==1) = mvnrnd(mu(1,:),sigma,length(X(states==1)))';
    X(:, states==2) = mvnrnd(mu(2,:),sigma,length(X(states==2)))';
    
    X = X';
    
    % Plot features
    %figure
    %scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
    %grid
    
    X = X';
    % Generate the true state sequence from a predefined (random) HMM
    % transition matrix multiplied by a logistic regression factor, 
    % using the true betas(defined above) and the feature vector X
    pi = generate_random_initial_dist(n_states);
    A = generate_random_transition_matrix(n_states); %uniform: ones(n_states)/n_states;

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
        
        for j=1:n_states
            or_transf_A(j,:,t) = repmat(exp(betas(:,j)'*[1;X(:,t)])/sum(exp(betas'*[1;X(:,t)])), 1, n_states);
        end       
        or_transf_A(:,:,t) = A.*or_transf_A(:,:,t)';
        or_transf_A(:,:,t) = normalize(or_transf_A(:,:,t));
        
        p = or_transf_A(state,:,t);
%         if(any(p~=p2'))
%             disp('ERRO');
%         end
        if t==1
            store_p(t,:)=pi;
        end
        if t~=T 
            store_p(t+1,:) = p;
        end
        state_seq(t) = state;
        
        state = sampleDiscrete(p');    
    end
     
    % Test with a state_seq which is a shifted version of the states that
    % generated X
    states=states';
    delay = 20; %20 samples
    state_seq = delayseq(states,delay);
    state_seq(state_seq==0) = states(1);
    
    obs_seq = poissrnd(lambdas(state_seq));
    %% RECOVER BETAS: using logitMn code (Newton method)
    % EM SUPERVISED METHOD: HAVING THE STATE SEQUENCE (BECAUSE IT IS
    % SUPERVISED) THE PARAMETERS (TRANSITION MATRIX AND INITIAL STATE
    % DISTRIBUTION) ARE OBTAINED BY SIMPLY COUNTING AND NORMALIZING
    
    % THE BETAS OF THE LOGISTIC REGRESSION (BETAS) ARE OBTAINED WITH THE
    % FEATURE VECTOR X AND THE STATE SEQUENCE ITSELF (IN THIS CASE ONLY,
    % BECAUSE WE HAVE ACCESS TO IT)
    
    clear P;
    %betas_est = mnrfit(X', state_seq);
    
    %SUPERVISED CASE: I HAVE ACCESS TO THE STATE SEQUENCE
    [model, llh] = logitMn(X, state_seq');
    %model.W contains the betas but the last row is the intercept, we will
    %change to put it as the first row
    W = model.W;
    W = [W(end,:);W];
    W = W(1:end-1,:);
    
    disp('Original betas')
    betas
    disp('Estimated betas')
    W
   
    X = X';
    % Plot features with the original and estimated decision boundaries
    figure
    %scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
    % Plot feature points from the first class
    scatter(X(states == 1,1), X(states == 1,2), 150, 'b', '.')
    % Plot feature points from the second class.
    hold on;
    scatter(X(states == 2,1), X(states == 2,2), 120, 'r', '.')
    %plot true decision boundary
    hold on
    plot_decision_boundary(betas);
    
    hold on
    %plot estimated decision boundary
    plot_decision_boundary(W);
    legend('Feature points from first class', 'Feature points from second class', 'True decision boundary', 'Estimated decision boundary')
    title(['Delay = ', num2str(delay), ' samples'])
    
    X = X';
    [y, P] = logitMnPred(model, X); %y is the predicted state sequence and P is the state posteriors according to the logistic regression model
    state_posterior = P;
    
    any(y~=states') %check if the estimated state sequence is equal to the true one (for the data X generation)
    
%     %%
%     %retrieving transformed betas of the reference class (mnrfit considers the reference class
%     %as the last one)
%     beta_est_transf = zeros(n_features+1,n_states);
%     state_posterior = zeros(T,n_states);
%     alpha = zeros(1,T);
%     for t=1:T
%         %for all classes
%         sum_ = 0;
%         for c=1:n_states-1 %except the last (reference) class
%             sum_ = sum_ + logit(X(:,t), c, betas_est);
%         end
%         
%         %probability of the reference class
%         state_posterior(t,end) = 1/(1+sum_);
%         for c=1:n_states-1
%             state_posterior(t,c) = state_posterior(t,end)*logit(X(:,t), c, betas_est);
%         end
%         alpha(t) = log(1/sum_);
%     end
%     X_ = [ones(1,T);X];
%     beta_est_transf(:,end) = alpha*pinv(X_);
%     
%     %retrieving transformed betas of the rest of the classes
%     for c=1:n_states-1
%         sum_ = 1;
%         for k=1:n_states-1
%             if(k~=c)
%                 sum_ = sum_ + logit(X(:,t), k, betas_est);
%             end
%         end
%         for t=1:T
%             alpha(t)= log(logit(X(:,t), c, betas_est)/sum_);
%         end
%         beta_est_transf(:,c) = alpha*pinv(X_);
%     end
%    
%     [~, est_state_seq] = max(state_posterior,[],2);
%     any(est_state_seq~=state_seq)
    

    %EM IS SIMPLY COUNTING AND NORMALIZING
    [estimated_pi, estimated_T] = analize_sequence_hmm(state_seq, 1, T, n_states, 'normalize');
    
    %state posterior is not enough to estimate the state sequence,it
    %multiplies by the state transitions
    
    
    est_p = zeros(T,n_states);
    
    for i=1:T
        if i==1
            %IF DELAY: 1; OTHERWISE, USE ESTIMATED_PI (estimated
            %initial state distribution from the supervised EM) or pi
            %(original initial state distribution)
            est_p(i,:) = estimated_pi;%1;%pi;
        else
            %IF DELAY: USE TRANSITION_MATRIX; OTHERWISE, USE ESTIMATED_T
            %(estimated transition matrix from the supervised EM) or A (original
            %transition matrix)
            est_p(i,:) = state_posterior(:,i).*estimated_T(state_seq(i-1),:)';%transition_matrix(state_seq(i-1),:)';%A(state_seq(i-1),:)';
        end
        
        est_p(i,:) = est_p(i,:)/sum(est_p(i,:));
    end
    
    [~, est_state_seq] = max(est_p,[],2);
    any(est_state_seq~=state_seq)
    
    
    figure
    subplot(5,1,1);
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
    subplot(5,1,2);
    plot(state_seq, 'b')
    lgd = legend('True state sequence HMM');
    lgd.Location = 'northeast';
    ylim([0.9 2.3])
    ylabel('State')
    subplot(5,1,3); 
    plot(y, 'r')
    ylim([0.9 2.3])
    lgd = legend('Estimated state sequence X');
    lgd.Location = 'northeast';
    ylabel('State')
    subplot(5,1,4); 
    plot(est_state_seq, 'y')
    ylim([0.9 2.3])
    lgd = legend('Estimated state sequence HMM');
    lgd.Location = 'northeast';
    ylabel('State')
    

%     or_transf_A = zeros(n_states, n_states, T);
%     %aux2 = zeros(n_states, n_states);
%     for t=1:T
%         %disp(['t=',num2str(t)])
%         %for j=1:n_states
%         %aux2(j,:) = repmat(exp(W(:,j)'*[1;X(:,t)])/sum(exp(W'*[1;X(:,t)])), 1, n_states);
%         %end
%         %aux = repmat(P(t,:)', 1,n_states);
%         %any(any(aux-aux2>=10^-10))
%         %aux
%         for j=1:n_states
%             or_transf_A(j,:,t) = repmat(exp(betas(:,j)'*[1;X(:,t)])/sum(exp(betas'*[1;X(:,t)])), 1, n_states);
%         end       
%         or_transf_A(:,:,t) = A.*or_transf_A(:,:,t)';
%         or_transf_A(:,:,t) = normalize(or_transf_A(:,:,t));
%     end
    
    est_A = zeros(n_states, n_states, T);
    %aux2 = zeros(n_states, n_states);
    for t=1:T
        %disp(['t=',num2str(t)]);
        %for j=1:n_states
        %aux2(j,:) = repmat(exp(W(:,j)'*[1;X(:,t)])/sum(exp(W'*[1;X(:,t)])), 1, n_states);
        %end
        %aux = repmat(P(t,:)', 1,n_states);
        %any(any(aux-aux2>=10^-10))
        %aux
        est_A(:,:,t) = estimated_T.*repmat(P(:,t)', n_states,1); %estimated_T em vez da original A
       
        est_A(:,:,t) = normalize(est_A(:,:,t));
    end
    
    %Viterbi for sequence decoding
    vmodel=struct;
    vmodel.At = est_A;
    est_lambdas = zeros(n_states,1);
    for i=1:n_states
        est_lambdas(i) = sum(obs_seq(state_seq==i))/length(obs_seq(state_seq==i));
    end
    vmodel.lambdas = est_lambdas;
    vmodel.s = estimated_pi;

    est_state_seq2 = hmmViterbi(obs_seq, vmodel);
    subplot(5,1,5); 
    plot(est_state_seq2, 'c')
    ylim([0.9 2.3])
    lgd = legend('Estimated state sequence HMM 2');
    lgd.Location = 'northeast';
    xlabel('t')
    ylabel('State')
    
    %check if the estimated state sequence is different than the true state
    %sequence
    any(est_state_seq2~=state_seq')
end


function P = normalize(A)
    [row, col] = size(A);
    for i=1:row
        %normalize by rows
       P(i,:)= A(i,:)/sum(A(i,:));
    end
end