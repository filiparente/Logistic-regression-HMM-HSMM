clear all;
rng(1); %fix seed for reproducibility

map=true;
% Parameters of the normal distributions (mean+std), one for each state/class
%IN 2D:
%
% random
 %mu = [-6 5; 10 -2; 1 -2; 1 5; 10 5;-6 -2];
 %mu = [-6 2; -5 4; -3 -1; 6 -2; 1 3; 5 -4];
 %sigma = [0.6 0.01]; % shared diagonal covariance matrix
%
% column 1's
 %mu = [3 0;-1.5 2.6; -1.5 -2.6];

%IN 3D:
mu = [0 2 0; 0 -2 0; 2 0 0; 0 0 2; 0 0 -2; -2 0 0];
sigma = [0.2 0.05 0.05]; % shared diagonal covariance matrix    

%betas = [2 1; 3 -3; 0 0];
%betas = [0.3 1 -0.2 -6; 0.5 3 -0.5 2; 0 0 0 0]; %R^(p+1)*k^2 , p=2, k=2, (3x4)
%betas = [-1 -0.5 0.7 0.9; -0.6 0.01 0.01 0.3; 0 0 0 0];
%betas = [0.8 -0.2 1 1; -2 1 2 -2; 0 0 0 0];
% IN HSMMS, THE DIAGONAL ENTRIES OF THE TRANSITION MATRIX MUST BE ZERO, AND
% THE NUMBER OF STATES MUST BE HIGHER THAN TWO SO THAT THE TRANSITION
% MATRIX IS DIFFERENT THAN THE IDENTITY, A MORE REALISTIC SETUP WITH A MORE
% COMPLEX ESTIMATION PROCESS
% THEREFORE, WE NEED TO FORCE BETAS(11,22,33,...,KK) TO BE -inf
%betas = [-inf -0.2 1 1 -inf -0.7 0.3 0.4 -inf; -inf 1 2 -2 -inf -0.4 0.6 -0.1 -inf; -inf 0 0 0 -inf 0 0 0 -inf];
% IN 2D:
% with random ones
 %betas = [0 -0.6 0.1 -0.2 0 0 0.1 -0.6 0;
 %         0 0.5 -0.2 -0.2 0 1 0.5 -0.2 0; 
 %         0 0 0 0 0 0 0 0 0];
%
% with 2 ones in each column
 %betas = [ 0 0 0; -0.75 1.3 0;-0.75 -1.3 0; 1.5 0 0; 0 0 0 ; -0.75 -1.3 0;1.5 0 0; -0.75 1.3 0; 0 0 0]';

% IN 3D: with 1 one in each element
betas = [ 0 0 0 0; 0 1 0 0; 0 -1 0 0; 1 0 0 0; 0 0 0 0; -1 0 0 0;0 0 -1 0; 0 0 1 0; 0 0 0 0]';

n_X_classes = 6;

transition_matrix = zeros(n_X_classes);
factor = 0.5;
transition_matrix(1,3) = factor;
transition_matrix(1,6) = factor;
transition_matrix(2,4) = factor;
transition_matrix(2,5) = factor;
transition_matrix(3,1) = factor;
transition_matrix(3,2) = factor;
transition_matrix(4,3) = factor;
transition_matrix(4,6) = factor;
transition_matrix(5,1) = factor;
transition_matrix(5,2) = factor;
transition_matrix(6,4) = factor;
transition_matrix(6,5) = factor;

%TO DO: GERAR X ATRAVES DE UM HSMM COM MATRIZ DE DURAÇAO 1 EM D=100 E 0 NO
%RESTO
%GARANTIR QUE O HSMM DO SISTEMA TEM S=[1 0 0] PARA COMEÇAR NO ESTADO 1


hsmmLR_vs_hsmm('C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\synthetic\run3\', 'original_lambdas', [10, 50, 100], 'betas', betas, 'mu', mu, 'sigma', sigma, 'n_states', 3, 'n_features', 3, 'n_X_classes', 6, 'length_dur', 5, 'dim', 2000, 'map', map, 'percentages', [0.9, 0.05, 0.05], 'transition_matrix', transition_matrix)

% Save originals and results
% keySet = {'n_states','length_dur','max_obs','original_lambdas', 'original_PAI', 'original_A', 'original_P', 'original_B', 'N', 'dim', 'Vk', 'obs_seq', 'state_seq', 'lambdas_est', 'PAI_est_final', 'A_est', 'B_est_final', 'P_est_final', 'Qest_final', 'lkh', 'll', 'elapsed_time',  'hit_rate', 'total_ll', 'iterations', 'max_iterations'};
% valueSet = {n_states, length_dur, max_obs, original_lambdas, original_PAI, original_A, original_P, original_B, N, dim, Vk, obs_seq, state_seq, lambdas_est, PAI_est_final, A_est, B_est_final, P_est_final, Qest_final, lkh, ll, elapsed_time, hit_rate, total_ll, ir, 500};
% iteration = iteration +1;
% 
% M = containers.Map(keySet,valueSet);
% save(['results_' num2str(iteration) '.mat'],'M');