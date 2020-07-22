%% A sequential pruning strategy for the selection 
%% of the numberof states in HMMs
function [model, store_k, store_pred_mse_dev, store_pred_mse_train, criterion, likelihood] = pruning_hmm(x, X, dev_obs_seq, kmin, kmax, tolerance, mode, map, max_iterations, init, output_dir)   
    %Start pruning
    k=kmax;
    
    pruning_models = cell(kmax-kmin+1,1);
    C = zeros(kmax-kmin+1,1);
        
    
    store_pred_mse_dev = zeros(kmax-kmin+1,1);
    store_pred_mse_train = zeros(kmax-kmin+1,1);
    criterion = zeros(kmax-kmin+1,1);
    likelihood = zeros(kmax-kmin+1,1);
    
    
    K = length(unique(x));
    dim = size(x,1);
    N = size(x,2);
    n_montecarlo = 10;
    store_k = zeros(n_montecarlo,3);
    init_ = struct;
    params = struct;
    criteria = {'aic', 'bic', 'mmdl'};
    
    %Initialization  
    for j=1:n_montecarlo
        k=kmax;
        [At_hmm,init_.B,init_.PAI,Vk,~,~, init_.lambdas]=hmmInitialize(x,k,K,dim*ones(N,1), true, init);
        params.A = At_hmm(:,:,1);

        init_.A = At_hmm(:,:,1);
        init_.Vk = Vk;

        
        for i=1:length(criteria)
            params.A = init_.A;
            params.B = init_.B;
            params.PAI = init_.PAI;
            params.lambdas = init_.lambdas;
            Vk = init_.Vk;

            criterion_ = criteria(i);
            k=kmax;
            n=1;
            while(k >= kmin)

                tic 

                %TRAIN: Run Baum Welch algorithm until some convergence criterion
                %is met
                [model, lkh, ir] = hmm_train(params, x', X, max_iterations, tolerance, mode, Vk);

                elapsed_time = toc

                %Get the set of estimated parameters with k states
                A_est = model.A;
                PAI_est = model.s';
                lambdas_est = model.lambdas;
                lambdas_tt = model.lambdas_tt;
                B_est = model.B;
                Qest = model.Qest;
                store_GAMMA = model.store_GAMMA;

                %compute total loglikelihood
                total_ll = lkh;%sum(ll,2);

                %store the model
                pruning_models{k-kmin+1,:}={A_est,PAI_est,lambdas_est, lambdas_tt, B_est, Qest, total_ll(end),ir, elapsed_time, store_GAMMA};

                %Store variables to plot 
                %prediction error (in dev and train)
                %vs. k pruning
                
                %Validation: mse of prediction dev with no Poisson fluctuations
                [store_pred_mse_dev(n), ~, ~, ~] = evaluate_hmm(model, dev_obs_seq, map, false, false);

                %Validation: mse of prediction train with no Poisson fluctuations
                [store_pred_mse_train(n), ~, ~, ~] = evaluate_hmm(model, x, map, model.s', false);




                %Compute stationary probability distribution
                % which is the left eigenvector of A associated with the unit eigen-value
                [V,D,W] = eig(A_est); %V are the right eigenvectors, D is the diagonal matrix with eigenvalues, whereas W contains the left eigenvectors

                [idx,eigenvalue] = find(abs(sum(D,2)-1)<1e-10);

                if eigenvalue==1
                    p_infty = W(:,idx);
                else
                    disp('ERROR. Cannot find stationary probability distribution of HMM.');
                end

                %Compute and store the value of the model selection criterion
                %MMDL
                if strcmp(criterion_, 'mmdl')
                    C(k-kmin+1) = total_ll(ir) - ((k^2)/2)*log(N*dim)-(1/2)*sum(log(N*dim*p_infty));
                elseif strcmp(criterion_, 'bic')
                %BIC
                    C(k-kmin+1) = total_ll(ir) - ((k^2+k)/2)*log(N*dim)
                elseif strcmp(criterion_, 'aic')
                %AIC
                    C(k-kmin+1) = total_ll(ir) - ((k^2+k)/2)
                end


                criterion(n) = C(k-kmin+1);
                likelihood(n) = total_ll(ir);
                
                n=n+1;
                %aux_model.lambdas = lambdas_est;
                %aux_model.A = A_est;
                %aux_model.store_ALPHA = store_ALPHA;
                %aux_model.P = P_est;
                %aux_model.PM = P_est;
                %aux_model.B = B_est;
                %aux_model.PAI = PAI_est;

                %Validation: mse of prediction
                %[mse, ~, predicted_count, ~] = evaluate_hsmm(aux_model, aux_model.A(:,:,end), dev_X, dev_obs_seq, map, x, false);
                %store_dev_mse(k-kmin+1) = mse;

                %likelihood of dev observation sequence with parameters trained
                %[~, lkh, ll, ir]=hsmm_new2(aux_model, dev_obs_seq, 0, size(dev_obs_seq,1)*ones(size(dev_obs_seq,2),1), tolerance, unique(dev_obs_seq), dev_X, [], [], mode);
                %store_dev_lkh(k-kmin+1) = lkh;

                %Find least probable state, i.e., the smallest element of p_infty
                [M,I] = min(p_infty);
                least_probable_state = I;

                %Prune the least probable state by deleting the corresponding
                %elements from A, PAI,lambdas and B
                A_est(:,least_probable_state) = [];
                A_est(least_probable_state,:) = [];

                %If it sums to 0, put uniform distribution
                if any(sum(A_est,2)==0)
                    [r,~] = find(sum(A_est,2) == 0);
                    A_est(r,:)=ones(length(r),k-1)/(k-2);	
                    A_est(r, r)=0;   
                end

                PAI_est(least_probable_state)=[];
                if sum(PAI_est)==0
                    PAI_est=ones(k-1,1)/(k-1);
                end
                lambdas_est(least_probable_state)=[];
                B_est(least_probable_state,:)=[];

                %Reduce the number of states
                k=k-1

                %Normalize again
                PAI_est = PAI_est/sum(PAI_est);
                A_est = normalize(A_est,2);


                clear params;

                %initial reduced model
                params.A = A_est;
                params.PAI=PAI_est;
                params.lambdas = lambdas_est;
                params.B = B_est;

            end

            %compute optimal state
            C= abs(C/(N*dim));
            [optimal_state,~] = find(min(C)==C);
            %optimal model is the one from the iteration corresponding to the optimal
            %state
            %optimal_model = pruning_models{optimal_state,:};
            optimal_state = optimal_state-1+kmin
            store_k(i,j) = optimal_state;
            
            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter(kmax:-1:kmin, store_pred_mse_dev, '.');
            hold on
            scatter(kmax:-1:kmin, store_pred_mse_train, '.');
            hold on
            lgd = legend({'HMM Pruning predictions dev', 'HMM Pruning predictions train'});
            lgd.Location = 'northeast';
            ylabel('MSE')
            xlabel('State');
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HMM_mse_predictions_' criterion_{:} '_' num2str(j) '.png']);
            saveas(gcf,[output_dir 'HMM_mse_predictions_' criterion_{:} '_' num2str(j) '.fig']);
            

            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter(kmax:-1:kmin, real(criterion), '.');
            hold on
            scatter(kmax:-1:kmin, likelihood, '.');
            lgd = legend({'HMM Pruning criterion', 'HMM Training Log-likelihood'});
            lgd.Location = 'northeast';
            xlabel('State')
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HMM_pruning_' criterion_{:} '_' num2str(j) '.png']);
            saveas(gcf,[output_dir 'HMM_pruning_' criterion_{:} '_' num2str(j) '.fig']);
            
            %optimal_A = optimal_model{1};
            %optimal_PAI = optimal_model{2};
            %optimal_lambdas = optimal_model{3};
            %optimal_lambdas_tt = optimal_model{4};
            %optimal_B = optimal_model{5};
            %optimal_Qest = optimal_model{6};
            %optimal_total_ll = optimal_model{7};
            %optimal_ir = optimal_model{8};
            %optimal_elapsed_time = optimal_model{9};
            %optimal_store_GAMMA = optimal_model{10};

            %clear model;

            %model.A = optimal_A;%A_est;
            %model.lambdas = optimal_lambdas;%lambdas_est;
            %model.lambdas_tt = optimal_lambdas_tt;
            %model.s = optimal_PAI;%PAI_est;
            %model.Qest = optimal_Qest;%Qest;
            %model.B = optimal_B; %B_est;
            %model.store_GAMMA = optimal_store_GAMMA;
            %model.total_ll = optimal_total_ll;
            %model.n_it = optimal_ir;
            %model.elapsed_time = optimal_elapsed_time;
            %model.n_states = optimal_state;
        end
    end

end
