function models = logit_poissonHsmmEm_new(x, X, dev_X, dev_obs_seq, map, prev_optimal_d, prev_optimal_PM, geometric, geometric_mode, K, dim, N, or_transf_A, or_lambdas, n_montecarlo, kmin_hmm, kmax_hmm, kmin_hsmm, kmax_hsmm, pnmin, pnmax, mode, init, output_dir)   
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
    % Input:
    %   X: n_features x n, feature vector
    %   x: 1 x n integer vector which is the sequence of observations
    %   init: model or n_states
    % Output:s
    %   model: trained model structure
    %   llh: loglikelihood
    % Written by Mo Chen (sth4nth@gmail.com).
%     n = size(x,1);
%     %d = max(x);
%     %X = sparse(x,1:n,1,d,n);
%     if isstruct(init)   % init with a model
%         A = init.A;
%         lambdas = init.lambdas;
%         s = init.s;
%         % TODO
%     elseif numel(init) == 1  % random init with latent k
%         if kmax_hsmm==kmin_hsmm
%             k = init;
%         else
%             k = kmax_hsmm;
%         end
%     end
    %matriz de transiçao inicial toda igual, ou random? tentei toda igual mas
    %os resultados nao foram muito bons, a matriz estimada tambem era quase
    %sempre a mesma, vou agora tentar random.
    %At = repmat(A, 1,1,n); %all transition matrix for all t are equal
    MT = dim*ones(N,1);
    
    tolerance = 1e-4;
    max_iterations = 100;
    
    %% PRUNING HMM
    if kmax_hmm~=kmin_hmm
        [model, store_k_hmm, store_pred_mse_dev_hmm, store_pred_mse_train_hmm, criterion_hmm, likelihood_hmm] = pruning_hmm(x, X, dev_obs_seq, kmin_hmm, kmax_hmm, tolerance, mode, map, max_iterations, init, output_dir);   
   
        %Plot graph: predictions mse train and dev vs. k pruning
        %figure;
        %scatter(kmax_hmm:-1:kmin_hmm, store_pred_mse_dev_hmm, '.');
        %hold on
        %scatter(kmax_hmm:-1:kmin_hmm, store_pred_mse_train_hmm, '.');
        %hold on
        %lgd = legend({'HMM Pruning predictions dev', 'HMM Pruning predictions train'});
        %lgd.Location = 'northeast';
        %ylabel('MSE')
        %xlabel('State')
        
        %figure;
        %scatter(kmax_hmm:-1:kmin_hmm, real(criterion_hmm(2:end)), '.');
        %hold on
        %scatter(kmax_hmm:-1:kmin_hmm, likelihood_hmm(2:end), '.');
        %lgd = legend({'HMM Pruning criterion', 'HMM Training Log-likelihood'});
        %lgd.Location = 'northeast';
        %xlabel('State')
        
        optimal_k_hmm = length(model.lambdas);
    else
        optimal_k_hmm = kmax_hmm;%states;
    end
    %optimal_k_hmm = 30;
    
    %% PRUNING HSMM
    if kmax_hsmm~=kmin_hsmm
        [model, store_k_hsmm, store_pred_mse_dev_hsmm, store_pred_mse_train_hsmm, criterion_hsmm, likelihood_hsmm] = pruning_hsmm(x, X, dev_obs_seq, kmin_hsmm, kmax_hsmm, tolerance, mode, map, max_iterations, or_transf_A, or_lambdas, init, output_dir); 
        
        %Plot graph: predictions mse train and dev vs. k pruning
        %figure;
        %scatter(kmax_hsmm:-1:kmin_hsmm, store_pred_mse_dev_hsmm, '.');
        %hold on
        %scatter(kmax_hsmm:-1:kmin_hsmm, store_pred_mse_train_hsmm, '.');
        %hold on
        %lgd = legend({'HSMM Pruning predictions dev', 'HSMM Pruning predictions train'});
        %lgd.Location = 'northeast';
        %ylabel('MSE')
        %xlabel('State')
        
        %figure;
        %scatter(kmax_hsmm:-1:kmin_hsmm, real(criterion_hsmm), '.');
        %hold on
        %scatter(kmax_hsmm:-1:kmin_hsmm, likelihood_hsmm, '.');
        %lgd = legend({'HSMM Pruning criterion', 'HSMM Training Log-likelihood'});
        %lgd.Location = 'northeast';
        %xlabel('State')
        
        optimal_k_hsmm = model.n_states;
        optimal_PM_est=model.P;
        models{1}.optimal_PM_est = optimal_PM_est;
        models{2}.optimal_PM_est = optimal_PM_est;
        optimal_d = floor(log(0.001)/log(1-min(optimal_PM_est)));
    else
        optimal_k_hsmm = kmax_hsmm;%states;
        optimal_d = prev_optimal_d;
        optimal_PM_est = prev_optimal_PM; 
    end
    
    %Fazer kavg e std fora do pruning, para cada critério, guardar numa estrutura
    %pruning_results = containers.Map;
    %pruning_results('k_hmm') = store_k_hmm;
    %pruning_results('k_hsmm') = store_k_hsmm;
    %pruning_results('kavg_hmm_aic') = mean(store_k_hmm(1,:));
    %pruning_results('kavg_hmm_bic') = mean(store_k_hmm(2,:));
    %pruning_results('kavg_hmm_mmdl') = mean(store_k_hmm(3,:));
    %pruning_results('kavg_hsmm_aic') = mean(store_k_hsmm(1,:));
    %pruning_results('kavg_hsmm_bic') = mean(store_k_hsmm(2,:));
    %pruning_results('kavg_hsmm_mmdl') = mean(store_k_hsmm(3,:));
    %pruning_results('kstd_hmm_aic') = std(store_k_hmm(1,:));
    %pruning_results('kstd_hmm_bic') = std(store_k_hmm(2,:));
    %pruning_results('kstd_hmm_mmdl') = std(store_k_hmm(3,:));
    %pruning_results('kstd_hsmm_aic') = std(store_k_hsmm(1,:));
    %pruning_results('kstd_hsmm_bic') = std(store_k_hsmm(2,:));
    %pruning_results('kstd_hsmm_mmdl') = std(store_k_hsmm(3,:)); 

    %save([output_dir, 'BERT_pruning_report.mat'], 'pruning_results');
    
    %optimal_k_hmm = optimal_k_hsmm;
    %optimal_k_hsmm = 16;
    %optimal_d=20;

    %optimal_state=16;
    %optimal_d = 10;
    %params.pn = 0.8;
    
    %n_montecarlo = 10; %run 10 times with different initialization
    
    n_models = 3; %tirei HMM-LR %4; %HMM, HMM-LR, HSMM, HSMM-LR
    best_mse = ones(n_models,1)*inf;
    store_mse = zeros(n_models, n_montecarlo);
    
    tuning = (pnmax~=pnmin);
    
    for n=1:n_montecarlo
        %Same initialization for both models HMM, HMM-LR     
        [At_hmm,params_hmm.B,params_hmm.PAI,Vk,~,K, params_hmm.lambdas] = hmmInitialize(x,optimal_k_hmm,K,dim*ones(N,1), kmax_hmm~=kmin_hmm, init); %if kmax==kmin means no prunning therefore P is a state duration matrix, if we want prunning then P is a geometric distribution where each state has one parameter

        %Same initialization for both models HSMM, HSMM-LR     
        [At,params.B,params.P,params.PAI,Vk,~,K, params.lambdas] = hsmmInitialize(x,optimal_k_hsmm,optimal_d,K,dim*ones(N,1), geometric, geometric_mode, init); %if geometric=0, P is a state duration matrix, if geometric=1 P is a geometric distribution where each state has one parameter
        if geometric
            if strcmp(geometric_mode,'normal') && optimal_PM_est
                params.P = optimal_PM_est;
            end
        end
        
        for i=1:n_models
            if i==1
                % HSMM-LR
                params.A = At;
                if n==1
                    if tuning
                        %% BINARY SEARCH FOR TUNING REGULARIZATION PARAMETER OF LOGISTIC REGRESSION
                        %Tune regularization parameter of Newton's method
                        left = pnmin;%1e-4;%1;%1e-4; %pnmin;
                        right = pnmax;%1;%pnmax;

                        nr=0;
                        checkpoints_ll = zeros(1,3);
                        checkpoints = zeros(1,3)
                        %seen_pn = [];
                        pn_models = cell(1,3);
                        max_n = 15;
                        store_dev_mse_pn = [];
                        %store_dev_lkh_pn = [];

                        while nr<max_n
                            [checkpoints, checkpoints_ll] = binary_search(left,right, checkpoints, checkpoints_ll);
                            
                           
                            if nr==0
                                j_=1:3;
                            else
                                j_=2;
                                if all(checkpoints==prev_checkpoints)
                                    %we are at same interval
                                    %change to second best interval
                                    
                                    left = checkpoints(find(checkpoints_ll==score(1)));
                                    right = checkpoints(find(checkpoints_ll==score(3))); 
                                    [checkpoints, checkpoints_ll] = binary_search(left,right, checkpoints, checkpoints_ll);

                                end
                            end
                            for j=j_
                                params.pn = checkpoints(j);
                                %if find(params.pn==seen_pn)
                                %    continue;
                                %end
                                [model, lkh, ll, ir]=hsmm_new2(params, x, max_iterations, MT, tolerance, Vk, X, or_transf_A, or_lambdas, mode, 'zeros');       
                                nr=nr+1;

                                %compute total loglikelihood
                                %total_ll = sum(ll,2);
                                %checkpoints_ll(i) = lkh;%total_ll(end); 

                                pn_models{j} = model;
                                %seen_pn = [seen_pn, params.pn];

                                %Validation: mse of prediction
                                [mse, ~, ~, ~] = evaluate_hsmm(model, model.A(:,:,end), dev_X, dev_obs_seq, map, x(end), false, false, false);
                                store_dev_mse_pn = [store_dev_mse_pn,mse];
                                checkpoints_ll(j) = mse;

                                %model.PAI = model.s;
                                %likelihood of dev observation sequence with parameters trained
                                %[~, lkh, ll, ir]=hsmm_new2(model, dev_obs_seq, 0, size(dev_obs_seq,1)*ones(size(dev_obs_seq,2),1), tolerance, unique(dev_obs_seq), dev_X, [], [], mode);
                                %store_dev_lkh_pn = [store_dev_lkh_pn, lkh];
                            end

                            %update
                            score = sort(checkpoints_ll);
                            left = checkpoints(find(checkpoints_ll==score(1)));
                            right = checkpoints(find(checkpoints_ll==score(2)));   
                            prev_checkpoints = checkpoints;
                            
                            if abs(left-right)<1e-2
                                break;
                            end
                        end

                        %model = pn_models{find(checkpoints==left)};
                        optimal_pn = checkpoints(find(min(checkpoints_ll)==checkpoints_ll));
                        params.pn = optimal_pn;
                        %params_hmm.pn = optimal_pn;
                    else
                        params.pn = pnmax;
                        optimal_pn = pnmax;
                    end
                end
                %optimal_pn = 0.9257;
                %params.pn = optimal_pn;
                %params_hmm.pn = optimal_pn;
            elseif i==2
                %HSMM
                params.A = At(:,:,1);
            elseif i==3
                %HMM : no zero diagonals
                params_hmm.A = At_hmm(:,:,1);

            else
                %HMM-LR : no zero diagonals
                params_hmm.A = At_hmm;
           
            end
                
             if i>=3 
                %Train HMM (i==3) or HMM-LR (i==4)
                [model, lkh, ir] = hmm_train(params_hmm, x', X, max_iterations, tolerance, mode, Vk);
                
                %Validation: mse of prediction
                [mse, prev_state_posteriors, predicted_count, out_A] = evaluate_hmm(model, dev_obs_seq, map, false, false);
         
      
            else
                %if i~=1 || n~=1
                    % Train HSMM and HSMM-LR
                    [model, lkh, ll, ir]=hsmm_new2(params, x, max_iterations, MT, tolerance, Vk, X, or_transf_A, or_lambdas, mode, 'zeros');           
                %end
                
                if size(model.P,2)~=optimal_d
                    disp('error');
                end
                
                %Validation: mse of prediction
                [mse, prev_state_posteriors, predicted_count, out_A] = evaluate_hsmm(model, model.A(:,:,end), dev_X, dev_obs_seq, map, x(end), false, false, false);
        
            end

            store_mse(i,n) = mse;
            if mse < best_mse(i)
                %Retrieve HSMM-LR model with k* and dmax* found from the Pruning
                %strategy with the HSMM model (to be faster) and pN* tuned in HSMM-LR 
                models{i} = model; 
                models{i}.pred_val = predicted_count;
                models{i}.prev_state_posteriors = prev_state_posteriors;
                models{i}.out_A = out_A;
            end
            clear model;
        end
    end
            
    %Report results and save mean+-std of predictions in the validation set along with the
    %corresponding model
    disp(['Optimal number of states after Pruning HSMM = ', num2str(optimal_k_hsmm), ' and duration ; Prunning HMM', num2str(optimal_k_hmm)]);
    disp(['Results after training with ', num2str(n_montecarlo), ' different initializations:']);
    disp('Mean +- std of predictions in validation set for model...');

    %HMM-LR
    %mean_hmmLR = mean(store_mse(4,:));
    %std_hmmLR = std(store_mse(4,:));
    %disp(['HMM-LR: ', num2str(mean_hmmLR), ' +- ', num2str(std_hmmLR)]);
    %models{4}.mean_pred_errors_val = mean_hmmLR;
    %models{4}.std_pred_errors_val = std_hmmLR;
    
    %HMM
    mean_hmm = mean(store_mse(3,:));
    std_hmm = std(store_mse(3,:));
    disp(['HMM: ', num2str(mean_hmm), ' +- ', num2str(std_hmm)]);
    models{3}.mean_pred_errors_val = mean_hmm;
    models{3}.std_pred_errors_val = std_hmm;
    models{3}.optimal_k = optimal_k_hmm;
    

    %HSMM
    mean_hsmm = mean(store_mse(2,:));
    std_hsmm = std(store_mse(2,:));
    disp(['HSMM: ', num2str(mean_hsmm), ' +- ', num2str(std_hsmm)]);
    models{2}.mean_pred_errors_val = mean_hsmm;
    models{2}.std_pred_errors_val = std_hsmm;
    models{2}.optimal_d = optimal_d;
    models{2}.optimal_k = optimal_k_hsmm;

    %HSMM-LR
    mean_hsmmLR = mean(store_mse(1,:));
    std_hsmmLR = std(store_mse(1,:));
    disp(['HSMM-LR: ', num2str(mean_hsmmLR), ' +- ', num2str(std_hsmmLR)]);
    models{1}.mean_pred_errors_val = mean_hsmmLR;
    models{1}.std_pred_errors_val = std_hsmmLR;
    models{1}.pn = optimal_pn;
    models{1}.optimal_d = optimal_d;
    models{1}.optimal_k = optimal_k_hsmm;

%% GRID SEARCH FOR TUNING REGULARIZATION PARAMETER OF LOGISTIC REGRESSION
% Tune regularization parameter of Newton's method
%
% optimal_pn = 0;
% best_ll = -inf;
% isli = 0; %number of Iterations Since Last Improvement
%
% for pn=pnmax:-1:pnmin
%    params.pn = pn;
%    if isli>10 && optimal_pn
%        break;
%    end
% 
%    tic 
%    [model, lkh, ll, ir]=hsmm_new2(params, x, max_iterations, MT, tolerance, Vk, X, or_transf_A, or_lambdas, mode);
%    elapsed_time = toc
% 
%    %compute total loglikelihood
%    total_ll = sum(ll,2);
% 
%    if total_ll(end) > best_ll
%        best_ll = total_ll(end);
%        best_model = model;
%        optimal_pn = pn;
%        isli = 0;
%    else
%        isli = isli+1;
%    end
% end
    
end


function [out_checkpoints, out_checkpoints_ll] = binary_search(left, right, checkpoints, checkpoints_ll)
    mid = (left + right) / 2; %ceil();
    
    out_checkpoints = sort([left, mid, right]);
    
    out_checkpoints_ll = zeros(size(checkpoints_ll));
    
    for i=1:3
        aux = find(out_checkpoints(i)==checkpoints);
        if aux
            out_checkpoints_ll(i) = checkpoints_ll(aux);
        end
    end
end

