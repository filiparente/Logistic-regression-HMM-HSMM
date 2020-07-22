profile on

prompt = 'Run nº... (accepted input: integer number 1,2,...) ?';
run = input(prompt);
init = 'smart';

data = 'BERT'; %'TFIDF'
geometric = true;
geometric_mode = 'normal'; %in the 'rand' mode, the geometric distribution is set to random in each different initialization, with the only constraint that it respects that the maximum duration (which is equal to the optimal_d found after Pruning) is not exceeded; in the 'normal' mode, the geometric distribution is always the same, despite different initializations, and equal to the geometric distribution found after Pruning.

n_features = 768;

percentages = [0.8,0.10,0.10];


if strcmp(data, 'BERT')
    path = 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\server\';
elseif strcmp(data, 'TFIDF')
    path = ['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features) '\'];
else
    disp('Data not recognized');
    return
end

normalize = true;
map = false;
mode = false; %if true, all features are used to estimate the weights of the
    % logistic regression; if false, only the features of the timestamps
    % associated with a transition, according to the state sequence
    % estimated by the EM algorithm.

%n_tot_states = 20; %[20,15,10,5,2];
%for i=1:length(n_tot_states)
%n_states=n_tot_states(i);

n_montecarlo = 10; %100

trainedmodel_path = false;%'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\1.0\trained models\com normalizacao\results2.mat'; %false; %'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\results\real\1.0\hsmmLR\sem_normalizacao\sampling\run1_results.mat';%false;%'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\results\real\1.0\hsmmLR\sem_normalizacao\sampling\run1_results.mat';
%run_hsmm(output_dir, 'trainedmodel_path', trainedmodel_path, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax', 10, 'kmin', 5, 'pruning', true);

%Predicted state sequence (knowning the observations) by the model using
%the parameters learned from the training phase
% dim=length(train_obs_seq); %test_obs_seq
% K = length(unique(train_obs_seq)); %test_obs_seq
% [~,~,~,~,Vk,~,K, ~] = hsmmInitialize(all_obs_seq,n_states,length_dur,K,dim*ones(N,1)); %test_obs_seq
% 
% B = zeros(n_states, length(Vk));
%  for i=1:n_states
%      %ss(i)=sum(log(1:i));
%      B(i,:) = poisspdf(Vk, model.lambdas(i))';
%      B(i,:)=B(i,:)/sum(B(i,:));
%  end        
%         
% [model, lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,lkh, ll, ir]=hsmm_new_(model.lambdas, model.s, model.A, B,model.P,all_obs_seq,0,length(all_obs_seq)*ones(N,1), 1e-10, Vk, all_X, or_transf_A, or_lambdas, model.betas); %test_obs_seq instead of all_obs_seq and test_X instead of all_X
% est_state_seq=model.Qest;
% figure
% plot(obs_seq, 'c') %test_obs_seq
% hold on
% plot(model.lambdas(est_state_seq), 'm')
% hold on
% lgd = legend({'True observation sequence HSMM', 'Estimated observation sequence HSMM'});
% lgd.Location = 'northeast';
% ylabel('State')
% xlabel('t')
% drawnow

%figure 
%plot(tot_points_x, tot_points_y);
%figure
%scatter(tot_points_x, tot_points_y);

%% Run BERT sentence transformer embeddings for multiple windows

dts = [1,3,4,6,12,24,48];%[3,4,6,12, 24,48]; %[1,3,4,6,12,24,48];
dws = []; %0,1,3,5,7
n_exampless = [4959, 1653, 1239, 826, 413, 206, 103]; %4959
subtract = [1,2,2,2];
trainedmodel_path = false;

%Pruning variables
kmax_hsmm = 15; %19
kmax_hmm = 15; %37
kmin_hsmm = 15;
kmin_hmm = 15;


for i=1:length(dts)
    dt = dts(i);
    n_examples = n_exampless(i);
    
    dataset_path = [path, num2str(dt),'.0\*.mat'];
    
    aux_output_dir = ['C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\', num2str(dt),'.0\'];
    if ~exist(aux_output_dir, 'dir')
       mkdir(aux_output_dir)
    end
    output_dir = [aux_output_dir, 'results\'];
    if ~exist(output_dir, 'dir')
       mkdir(output_dir)
    end
    output_dir = [output_dir, data, '\'];
    if ~exist(output_dir, 'dir')
       mkdir(output_dir)
    end
    output_dir = [output_dir, 'run', num2str(run), '\'];
    if ~exist(output_dir, 'dir')
       mkdir(output_dir)
    end
    %trainedmodel_path = 'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\1.0\results\results2.mat'; %false;
    
    %Pruning for k and dmax and tuning for pN is done at the dt level
    %models = run_hsmm(output_dir, 'trainedmodel_path', trainedmodel_path, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax_hmm', kmax_hmm,'kmax_hsmm', kmax_hsmm, 'kmin_hmm', kmin_hmm, 'kmin_hsmm', kmin_hsmm, 'pruning', true,  'tuning', true, 'pnmax', 1, 'pnmin', 1e-4, 'n_examples', n_examples, 'mode', mode);
    %if dt==1
    %    models = run_hsmm(output_dir, 'geometric', geometric, 'geometric_mode', geometric_mode, 'dataset_path', dataset_path,'trainedmodel_path', trainedmodel_path, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax_hmm', 36,'kmax_hsmm', 16, 'kmin_hmm', 36, 'kmin_hsmm', 16,  'optimal_d_hsmm', 21, 'pruning', false,  'tuning', false, 'pnmax', 0.9297, 'pnmin', 0.9297, 'n_examples', n_examples, 'mode', mode);
    %    %models = load('C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\1.0\results\BERT\models.mat');
    %    %models = models.models;
    %elseif dt==3
    %    models = run_hsmm(output_dir, 'geometric', geometric, 'geometric_mode', geometric_mode, 'dataset_path', dataset_path, 'trainedmodel_path', trainedmodel_path, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax_hmm', 50,'kmax_hsmm', 12, 'kmin_hmm', 50, 'kmin_hsmm', 12,  'optimal_d_hsmm', 19, 'pruning', false,  'tuning', false, 'pnmax', 0.9688, 'pnmin', 0.9688, 'n_examples', n_examples, 'mode', mode);  
    %else
    %models = load('C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\1.0\results\BERT\run1\models.mat');
    %optimal_PM_est = models.models.hsmmLr_model.PM;
    models = run_hsmm(output_dir, 'percentages', percentages, 'geometric', geometric, 'geometric_mode', geometric_mode, 'dataset_path', dataset_path, 'trainedmodel_path', trainedmodel_path, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax_hmm', kmax_hmm,'kmax_hsmm', kmax_hsmm, 'kmin_hmm', kmin_hmm, 'kmin_hsmm', kmin_hsmm, 'optimal_d_hsmm', 20, 'optimal_PM_est',  0, 'pruning', true,  'tuning', true, 'pnmax', 1, 'pnmin', 1e-4, 'n_examples', n_examples, 'mode', mode, 'init', init);
    %models = load('C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\1.0\results\TFIDF\run6\models.mat');
    %models=models.models;
    %end
    
    
    
    %Use the tuned parameters for every window
    %The windows only change the features so we only need to evaluate our
    %model HSMM-LR. The results for the others are the same.
    optimal_k_hsmm = models.hsmm_model.optimal_k;
    kmax_hsmm = optimal_k_hsmm+5;%round(0.1*(kmax_hsmm-optimal_k_hsmm));
    
    optimal_d_hsmm = models.hsmm_model.optimal_d;
    if geometric
        if isfield(models.hsmmLr_model, 'optimal_PM_est')
            optimal_PM_est = models.hsmmLr_model.optimal_PM_est;
        else
            optimal_PM_est = models.hsmm_model.PM;
        end
    else
        optimal_PM_est = 0;
    end
    
    optimal_k_hmm = models.hmm_model.optimal_k;
    kmax_hmm = optimal_k_hmm+round(0.1*(kmax_hmm-optimal_k_hmm));
    clear dataset_path;
    
    for j=1:length(dws)
        dw=dws(j);
        n_examples = n_examples-subtract(j);
        if strcmp(data,'TFIDF')
            n_examples = n_examples-2*dw;
        end
        
        dataset_path = [path, num2str(dt),'.', num2str(dw),'\*.mat'];
        aux_output_dir = ['C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\', num2str(dt),'.', num2str(dw),'\'];
        if ~exist(aux_output_dir, 'dir')
           mkdir(aux_output_dir)
        end
        output_dir = [aux_output_dir, 'results\'];
        if ~exist(output_dir, 'dir')
           mkdir(output_dir)
        end
        output_dir = [output_dir, data, '\'];
        if ~exist(output_dir, 'dir')
           mkdir(output_dir)
        end

        output_dir = [output_dir, 'run', num2str(run), '\'];
        if ~exist(output_dir, 'dir')
           mkdir(output_dir)
        end
    
        
        run_hsmm(output_dir,'percentages', percentages,  'geometric', geometric, 'geometric_mode', geometric_mode, 'dataset_path', dataset_path, 'trainedmodel_path', false, 'normalize', normalize, 'n_montecarlo', n_montecarlo, 'map', map, 'kmax_hmm', optimal_k_hmm, 'kmax_hsmm', optimal_k_hsmm, 'kmin_hmm', optimal_k_hmm, 'kmin_hsmm', optimal_k_hsmm, 'optimal_d_hsmm', optimal_d_hsmm, 'optimal_PM_est', optimal_PM_est, 'pruning', false, 'tuning', true, 'pnmax', 1, 'pnmin', 1e-4, 'n_montecarlo', n_montecarlo, 'n_examples', n_examples, 'mode', mode, 'init', init);
        
        clear dataset_path;
    end
    
    profile viewer
end
        




    