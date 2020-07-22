%%
%clear all

keySet = {'n_states','length_dur','max_obs','original_lambdas', 'original_PAI', 'original_A', 'original_P', 'original_B', 'N', 'dim', 'Vk', 'obs_seq', 'state_seq', 'lambdas_est', 'PAI_est_final', 'A_est', 'B_est_final', 'P_est_final', 'Qest_final', 'lkh', 'll', 'elapsed_time',  'hit_rate', 'total_ll'};

%Define the number of states of the markov chain
n_states = 3;

%Define the maximum length of state duration
length_dur = 4;

%Max observable value
max_obs = 20;

%Create the markov modulated poisson process original model
original_lambdas = [2, 15, 30];

rng(1);
%Initialize the state initial distribution vector of the markov chain with random values
original_PAI = generate_random_initial_dist(n_states);%ones(n_states,1)*(1/n_states);

rng(1);
%Initialize the transition matrix of the markov chain with random values
original_A = generate_random_transition_matrix_d(n_states); %[0,    1/(n_states-1),    1/(n_states-1);1/(n_states-1),    0,    1/(n_states-1);1/(n_states-1),    1/(n_states-1),    0];


%for d=1:10
%rng(1);
%original_P = zeros(n_states, length_dur);
%original_P(:,d)=1;%[0, 0, 0.5, 0.5; 0, 0, 0.5, 0.5; 0, 0, 0.5, 0.5];%generate_random_dur_dist(n_states, length_dur);
%% REPLACED original_P = ones(n_states,length_dur)*(1/length_dur); BY
original_P = generate_random_dur_dist(n_states, length_dur);
%original_PM = [0.1, 0.2, 0.5];

%for dim=10:500:10000
rng(1);
original_B = generate_random_dur_dist(n_states, max_obs);
%original_B = [0.05, 0.9, 0.05; 0.9, 0.05, 0.05; 0, 0.1, 0.9];

%Number of observations
N = 1;

%Dimension of the observations
dim = 100;

%all observations from the 100 monte carlo runs
total_obs_seq = zeros(100);
total_state_seq = zeros(100);

iteration = 0; 

% rng(1);
% [Vk, obs_seq, state_seq] = hsmmSample(original_PAI,original_A,original_P,original_B, original_lambdas, dim,N);
% K= sum(unique(Vk));
% if N~=1
%     x=cell2mat(obs_seq)';
% else
%     x=obs_seq;
% end
% 
% rng(1);
% [A,B,P,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x',n_states,length_dur,K,dim*ones(1,1));
% 
% Vk = unique(Vk);
% if sum(Vk==0)>0
%     Vk=Vk(2:end);
% end

%Iterate reducing sequence length
%while(dim >= 10)
%save('originals_dmax_test.mat', 'length_dur', 'n_states', 'max_obs', 'original_A', 'original_P', 'original_B', 'original_lambdas', 'original_PAI', 'N', 'dim');
d=2;
%for d=2:2:6
    %dim = dim/10;
    results = cell(100, 12);
    for iter=1:100 %repeat 25 times for statistical significance
        %rng(1);
        %% REPLACED 
        [Vk, obs_seq, state_seq] = hsmmSample(original_PAI,original_A,original_P,original_B, original_lambdas, dim,N);
        %[Vk, obs_seq, state_seq] = hsmmSample(original_PAI,original_A,original_PM,original_B, original_lambdas, dim,N);
        
        K = sum(unique(Vk));
        if N~=1
            x=cell2mat(obs_seq)';
            s=cell2mat(state_seq)';
        else
            x=obs_seq';
            s=state_seq';
        end
        
        total_obs_seq(:,iter)=x;
        total_state_seq(:, iter)=s;
        
    end
    
    
    save('C:\Users\Filipa\Desktop\tese\other\file.mat', 'total_obs_seq', 'total_state_seq', 'length_dur', 'n_states', 'max_obs', 'original_A', 'original_P', 'original_B', 'original_lambdas', 'original_PAI', 'N', 'dim', '-v7.3');
 
 %%
 file = open('C:\Users\Filipa\Desktop\tese\other\file.mat');
    
   
    
    
    %assert sample is OK! estimated_pi must be close to original_pi,
        %estimated_T to original_A and estimated_D to original_P!

        %[estimated_pi, estimated_T, estimated_D] = analize_sequence(cell2mat(state_seq)', n_states, length_dur)

        %obs_seq = obs_seq';
        %state_seq = state_seq' +1;
         %rng(1);
        %% REPLACED 
        [A,B,P,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,n_states,d,K,dim*ones(N,1));
        %[A,B,PM,PAI,Vk,MO,K, lambdas] = hsmmInitialize(x,n_states,d,K,dim*ones(N,1));

         %Vk = unique(Vk);
         %if sum(Vk==0)>0
         %    Vk=Vk(2:end);
         %end

        tic
        %rng(1);
        %% REPLACED 
        [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,P,x,100,dim*ones(N,1), 1e-10, Vk);
        %[lambdas_est, lambdas_tt, PAI_est,A_est,B_est,PM_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, x,100,dim*ones(N,1), 1e-100, Vk);
        %[~,~,~,~,~,lkh1, ll1]=hsmm_new(original_PAI,original_A,original_B,original_P,obs_seq',0,dim*ones(N,1), 1e-10);
        elapsed_time = toc

        
        %Find the optimal assgnment of states
        [assignment, total_cost] = munkres2(n_states, n_states, s, Qest);

        %Compute final estimated state sequence according to the assignment
        Qest_final = zeros(size(Qest));
        for i=1:length(assignment)
            Qest_final(Qest==i) = assignment(i);
        end

        %Compute final parameters according to the assignment
        %With the indexes association, permute the rows and columns of the predicted 
        %transition matrix so that it matches the true transition matrix
        %Same for the duration probability density function (but only rows, since rows are states and columns are durations)
        B_est_final = zeros(size(B_est));
        P_est_final = zeros([n_states, d]);
        PAI_est_final = zeros(n_states,1);
        lambdas_est_final = zeros(1, n_states);
        prev_assign = [];
        for i=1:length(assignment)
            P_est_final(assignment(i),:) = P_est(i,:);
            B_est_final(assignment(i),:) = B_est(i,:);
            PAI_est_final(assignment(i)) = PAI_est(i);
            lambdas_est_final(assignment(i)) = lambdas_est(i);
            [v,idx] = find(i==prev_assign(:));

            if (~isempty(v) && assignment(i) == v) || i==assignment(i) %the swap was already performed
                prev_assign = [prev_assign, assignment(i)];
                continue;
            else
                A_est([i, assignment(i)], :) = A_est([assignment(i),i], :);     % swap rows.
                A_est(:, [i, assignment(i)]) = A_est(:, [assignment(i),i]);     % swap columns.
                prev_assign = [prev_assign, assignment(i)];
            end 
        end 

        hit_rate = (sum(sum(s == Qest_final))/(N*dim));
        fprintf('Percentage of right state estimates %.2f %%\n', hit_rate*100);

        % plot(ll);
        % figure
         total_ll = sum(ll,2);
        % figure
        % plot(total_ll);
        % figure
        % plot3(1:1:ir, sum(diff(lambdas_tt')), ll, '+');
        % for i=1:N
        %     figure
        %     plot(state_seq{i}, 'r');
        %     hold on
        %     plot(Qest_final(:,i),'b');
        % end

        for i=1:1%N
            figure
            plot(x(:,i));
            hold on
            plot(lambdas_est(Qest_final(:,i)));
        end
        results{iter,1} = lambdas_est_final';
        results{iter,2} = PAI_est_final;
        results{iter,3} = A_est;
        results{iter,4} = B_est_final;
        results{iter,5} = P_est_final;
        results{iter,6} = Qest_final;
        results{iter,7} = lkh;
        results{iter,8} = ll;
        results{iter,9} = elapsed_time;
        results{iter,10} = hit_rate;
        results{iter,11} = total_ll;
        results{iter,12} = ir;
    end

     keySet = {'n_states','length_dur','max_obs','original_lambdas', 'original_PAI', 'original_A', 'original_P', 'original_B', 'N', 'dim', 'Vk', 'obs_seq', 'state_seq', 'mean_lambdas_est', 'mean_PAI_est', 'mean_A_est', 'mean_P_est', 'mean_hit_rate' ,'max_iterations', 'mean_elapsed_time'};
     valueSet = {n_states, length_dur, max_obs, original_lambdas, original_PAI, original_A, original_P, original_B, N, dim, Vk, obs_seq, state_seq, mean_matrices({results{:,1}}), mean_matrices({results{:,2}}), mean_matrices({results{:,3}}), mean_matrices({results{:,5}}), mean_matrices({results{:,10}}), 100, mean([results{:,9}])};
     iteration = iteration +1;
    
     M = containers.Map(keySet,valueSet);
     save(['results_dmax=' num2str(d) '.mat'],'M');
     clear M;
%end
%end
end
%% ANALYSIS PART
k=1;
for i=10:500:5010
    %data structure
    S(k) = load(['results_dim=' num2str(i) '.mat'], 'M');    
    %S(i) = load(['results_dur_d=' num2str(i) '.mat'], 'M');
    k=k+1;
    
end


% mse A
for i=1:11
    tmp = (S(i).M('original_A') - S(i).M('mean_A_est')).^2;
    MSE_A(i) = sum(tmp(:))/numel(S(i).M('original_A'));
    
    tmp = (S(i).M('original_P') - S(i).M('mean_P_est')).^2;
    MSE_P(i) = sum(tmp(:))/numel(S(i).M('original_P'));
    
    tmp = (S(i).M('original_PAI') - S(i).M('mean_PAI_est')).^2;
    MSE_PAI(i) = sum(tmp(:))/numel(S(i).M('original_PAI'));
    
    tmp = (S(i).M('original_lambdas') - S(i).M('mean_lambdas_est')').^2;
    MSE_lambdas(i) = sum(tmp(:))/numel(S(i).M('original_lambdas'));
    
    hit_rate(i) = S(i).M('mean_hit_rate');
end


%%
%plots
figure
plot(1:11, hit_rate)
ylabel('Hit rate')
xlabel('Duration')

figure
plot(1:11, MSE_lambdas)
ylabel('MSE lambdas')
xlabel('Duration')

figure
plot(1:11, MSE_PAI)
ylabel('MSE PAI')
xlabel('Duration')

figure
plot(1:11, MSE_P)
ylabel('MSE P')
xlabel('Duration')

figure
plot(1:11, MSE_A)
ylabel('MSE A')
xlabel('Duration')