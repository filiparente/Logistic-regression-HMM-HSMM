%% A sequential pruning strategy for the selection 
%% of the numberof states in HSMMs
function [model, store_k, store_d, store_pred_mse_dev, store_pred_mse_train, criterion, likelihood] = pruning_hsmm(x, X, dev_obs_seq, kmin, kmax, tolerance, mode, map, max_iterations, or_transf_A, or_lambdas, init, output_dir)   
 
           
    %Start pruning
    k=kmax;
    
    %store_k = zeros(kmax-kmin+1,1);
    n_montecarlo = 10;
    store_k = zeros(n_montecarlo,3);
    store_d = zeros(n_montecarlo,3);
    store_pred_mse_dev = zeros(kmax-kmin+1,1);
    store_pred_mse_train = zeros(kmax-kmin+1,1);
    store_d_aux = zeros(kmax-kmin+1,1);
    criterion = zeros(kmax-kmin+1,1);
    likelihood = zeros(kmax-kmin+1,1);
    n=1;
    criteria = {'aic', 'bic', 'mmdl'};
    
    
    K = length(unique(x));
    dim = size(x,1);
    N = size(x,2);
    
    for g=1:n_montecarlo
         k=kmax;
        [At,init_.B,init_.P,init_.PAI,Vk,~,~, init_.lambdas] = hsmmInitialize(x,k,48,K,dim*ones(N,1), true, 'rand', init); 

        params.A = At(:,:,1);
        init_.A = At(:,:,1);
        init_.Vk = Vk;
        
        for z=1:length(criteria)
            models = {};
            pruning_models = cell(kmax-kmin+1,1);
            C = zeros(kmax-kmin+1,1);

            MT = dim*ones(N,1);

            params.A = init_.A;
            params.B = init_.B;
            params.PAI = init_.PAI;
            params.P = init_.P;
            params.lambdas = init_.lambdas;
            Vk = init_.Vk;

            criterion_ = criteria(z);
            k=kmax;
            n=1;
            while(k >= kmin)

                tic 
                %% REPLACED [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,P,x,100,dim*ones(N,1), 1e-10, Vk); BY
                %[lambdas_est, lambdas_tt, PAI_est,A_est,B_est,PM_est,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, x,100,dim*ones(N,1), 1e-100, Vk);

                %if m==1
                %    [model, lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest, store_GAMMA, store_ALPHA, lkh, ll, ir]=hsmm_new_(lambdas, PAI,At,B,P,x,100,dim*ones(N,1), 1e-20, Vk, X, or_transf_A, or_lambdas);
                %else
                %    [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest, store_GAMMA, store_ALPHA, lkh, ll, ir]=hsmm_new(lambdas, PAI,At(:,:,1),B,P,x,100,dim*ones(N,1), 1e-10, Vk);
                %end
                [model, lkh, ll, ir]=hsmm_new2(params, x, max_iterations, MT, tolerance, Vk, X, or_transf_A, or_lambdas, mode);

                elapsed_time = toc

                A_est = model.A;
                PAI_est = model.s;
                lambdas_est = model.lambdas;
                lambdas_tt = model.lambdas_tt;
                B_est = model.B;
                P_est = model.P;
                Qest = model.Qest;
                store_GAMMA = model.store_GAMMA;
                store_ALPHA = model.store_ALPHA;
                if isfield(model, 'betas')
                    betas_est = model.betas;
                end

                %compute total loglikelihood
                total_ll = sum(ll,2);


                %store the model
                %% REPLACED models{k-kmin+1,:}={A_est,PAI_est,lambdas_est,B_est,P_est, Qest, total_ll(end)}; BY    
                if isfield(model, 'PM')
                    PM_est = model.PM;
                    geometric = true;
                    pruning_models{k-kmin+1,:}={A_est,PAI_est,lambdas_est,lambdas_tt, B_est,PM_est, Qest, total_ll(end),ir, elapsed_time, store_GAMMA, store_ALPHA};

                else
                    geometric = false;
                    pruning_models{k-kmin+1,:}={A_est,PAI_est,lambdas_est,lambdas_tt, B_est,P_est, Qest, total_ll(end),ir, elapsed_time, store_GAMMA, store_ALPHA};
                end

                %Store variables to plot 
                %prediction error (in dev and train)
                %vs. k pruning
                %store_k(n) = k;

                %Validation: mse of prediction dev with no Poisson fluctuations
                [store_pred_mse_dev(n), ~, ~, ~] = evaluate_hsmm(model, model.A(:,:,end), [], dev_obs_seq, map, x(end), false, false, false);

                %Validation: mse of prediction train with no Poisson fluctuations
                [store_pred_mse_train(n), ~, ~, ~] = evaluate_hsmm(model, model.A(:,:,end), [], x, map, x(1), repmat(model.s,1,size(model.P,2)), false, false);



                %estimator of p_infty
                jumps = cell(N,1);
                counting_process_NI = zeros(N,k);
                counting_process_N = zeros(N,1);
                for i=1:N
                    jumps{i} = [Qest(1,i); Qest([0;diff(Qest(:,i))~=0]>0,i)];
                    counting_process_N(i) = length(jumps{i})-1;
                    for j=1:k
                        %number of visits of the semi markov chain to state j up to time T
                        %in the ith observation sequence
                        counting_process_NI(i,j) = sum(jumps{i}==j);
                    end
                end

                %sum the counting process for all obs sequences ???
                counting_process_NI = sum(counting_process_NI, 1);


        %             if geometric
        %                 if m==1
        %                     PM_est = model.PM;
        %                     P_est = model.P;
        %                     d=[1:1:size(P_est,2)];
        %                 else
        %                     PM_est=P_est;
        %                     clear P_est;
        %                     length_dur = floor(log(0.001)/log(1-min(PM_est)));
        %                     d=[1:1:length_dur];
        %                     for aux=1:k
        %                         P_est(aux,:) = PM_est(aux)*(1-PM_est(aux)).^(d-1); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
        %                         P_est(aux,:)=P_est(aux,:)/sum(P_est(aux,:));
        %                     end
        %                 end
        %             end
                p_infty_ = (counting_process_NI/sum(counting_process_N))';
                expected_dur=zeros(1,k);
                d=[1:1:size(P_est,2)];
                for i=1:k
                    expected_dur(i)=P_est(i,:)*d';
                end
                %p_infty_hsmm = p_infty.*m';
                %p_infty_hsmm = p_infty_hsmm/sum(p_infty_hsmm);
                %or
                p_infty_hsmm2 = p_infty_.*expected_dur';
                p_infty_hsmm2 = p_infty_hsmm2/sum(p_infty_hsmm2);

                %MMDL
                if strcmp(criterion_, 'mmdl')
                    C(k-kmin+1) = total_ll(ir) - ((k^2-k)/2)*log(N*dim)-sum(log(N*dim*p_infty_hsmm2));
                elseif strcmp(criterion_, 'bic')
                %BIC
                    C(k-kmin+1) = total_ll(ir) - ((k^2+k)/2)*log(N*dim);
                elseif strcmp(criterion_, 'aic')
                %AIC
                    C(k-kmin+1) = total_ll(ir) - ((k^2+k)/2);
                end
                criterion(n) = C(k-kmin+1);

                likelihood(n) = total_ll(ir);
                store_d_aux(n) = floor(log(0.001)/log(1-min(PM_est)));

                n=n+1;

                aux_model.lambdas = lambdas_est;
                aux_model.A = A_est;
                aux_model.store_ALPHA = store_ALPHA;
                aux_model.P = P_est;
                aux_model.PM = P_est;
                aux_model.B = B_est;
                aux_model.PAI = PAI_est;

                %Validation: mse of prediction
                %[mse, ~, predicted_count, ~] = evaluate_hsmm(aux_model, aux_model.A(:,:,end), dev_X, dev_obs_seq, map, x, false);
                %store_dev_mse(k-kmin+1) = mse;

                %likelihood of dev observation sequence with parameters trained
                %[~, lkh, ll, ir]=hsmm_new2(aux_model, dev_obs_seq, 0, size(dev_obs_seq,1)*ones(size(dev_obs_seq,2),1), tolerance, unique(dev_obs_seq), dev_X, [], [], mode);
                %store_dev_lkh(k-kmin+1) = lkh;

                %compute least probable state
                [M,I] = min(p_infty_hsmm2);
                least_probable_state = I;

                %prune
                size_A = size(A_est);
                if size_A(end)==k %HSMM, stationary A
                    A_est(:,least_probable_state) = [];
                    A_est(least_probable_state,:) = [];

                    if any(sum(A_est,2)==0)
                        [r,~] = find(sum(A_est,2) == 0);
                        A_est(r,:)=ones(length(r),k-1)/(k-2);	
                        A_est(r, r)=0;   
                    end
                else
                    A_est(:,least_probable_state,:) = [];
                    A_est(least_probable_state,:,:) = [];

                    if any(any(sum(A_est,2)==0))
                        [r,~,v] = ind2sub(size(sum(A_est,2)),find(sum(A_est,2) == 0));
                        A_est(r,:,v)=ones(length(r),k-1)/(k-2);	
                        A_est(r, r, v)=0;   
                    end
                end

                PAI_est(least_probable_state)=[];
                lambdas_est(least_probable_state)=[];
                B_est(least_probable_state,:)=[];
                %% REPLACED P_est(least_probable_state,:)=[]; BY
                if geometric
                    PM_est(least_probable_state) = [];
                    P_est = PM_est;
                else
                    P_est(least_probable_state,:)=[];
                end

                k=k-1;
                %normalize again
                PAI_est = PAI_est/sum(PAI_est);
                %for j=1:k
                %    A_est(j,:)=A_est(j,:)/sum(A_est(j,:));
                %end
                A_est = normalize(A_est,2);


                clear params;

                %initial reduced model
                %if m==1
                %    params.A=A_est;
                %else
                %    params.A = repmat(A_est, dim);
                %end
                params.A = A_est;
                params.PAI=PAI_est;
                params.lambdas = lambdas_est;
                params.B = B_est;
                %% REPLACED P = P_est; BY
                params.P = P_est;

            end

            %compute optimal state
            C= abs(C/(N*dim));
            [optimal_state,~] = find(min(C)==C);
            %optimal model is the one from the iteration corresponding to the optimal
            %state
            optimal_model = pruning_models{optimal_state,:};
            optimal_state = optimal_state-1+kmin
            store_k(z,g) = optimal_state;
            
            store_d(z,g) = floor(log(0.001)/log(1-min(optimal_model{6})));
            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter(kmax:-1:kmin, store_pred_mse_dev, '.');
            hold on
            scatter(kmax:-1:kmin, store_pred_mse_train, '.');
            hold on
            lgd = legend({'HSMM Pruning predictions dev', 'HSMM Pruning predictions train'});
            lgd.Location = 'northeast';
            ylabel('MSE')
            xlabel('State');
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HSMM_mse_predictions_' criterion_{:} '_' num2str(g) '.png']);
            saveas(gcf,[output_dir 'HSMM_mse_predictions_' criterion_{:} '_' num2str(g) '.fig']);
            
            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter3(kmax:-1:kmin, store_d_aux, store_pred_mse_dev, '.');
            hold on
            scatter3(kmax:-1:kmin, store_d_aux, store_pred_mse_train, '.');
            hold on
            lgd = legend({'HSMM Pruning predictions dev', 'HSMM Pruning predictions train'});
            lgd.Location = 'northeast';
            ylabel('Maximum duration')
            xlabel('State');
            zlabel('MSE');
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HSMM_mse_predictions3d_' criterion_{:} '_' num2str(g) '.png']);
            saveas(gcf,[output_dir 'HSMM_mse_predictions3d_' criterion_{:} '_' num2str(g) '.fig']);
            

            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter(kmax:-1:kmin, real(criterion), '.');
            hold on
            scatter(kmax:-1:kmin, likelihood, '.');
            lgd = legend({'HSMM Pruning criterion', 'HSMM Training Log-likelihood'});
            lgd.Location = 'northeast';
            xlabel('State')
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HSMM_pruning_' criterion_{:} '_' num2str(g) '.png']);
            saveas(gcf,[output_dir 'HSMM_pruning_' criterion_{:} '_' num2str(g) '.fig']);
            
            FigH = figure('Position', get(0, 'Screensize'),'visible','on');
            scatter3(kmax:-1:kmin, store_d_aux, real(criterion), '.');
            hold on
            scatter3(kmax:-1:kmin, store_d_aux, likelihood, '.');
            lgd = legend({'HSMM Pruning criterion', 'HSMM Training Log-likelihood'});
            lgd.Location = 'northeast';
            xlabel('State')
            ylabel('Maximum duration')
            F=getframe(FigH);

            saveas(gcf,[output_dir 'HSMM_pruning3d_' criterion_{:} '_' num2str(g) '.png']);
            saveas(gcf,[output_dir 'HSMM_pruning3d_' criterion_{:} '_' num2str(g) '.fig']);
            
            %optimal_A = optimal_model{1};
            %optimal_PAI = optimal_model{2};
            %optimal_lambdas = optimal_model{3};
            %optimal_lambdas_tt = optimal_model{4};
            %optimal_B = optimal_model{5};
            %optimal_PM = optimal_model{6};
            %optimal_d = size(optimal_PM,2);
            %optimal_Qest = optimal_model{7};
            %optimal_total_ll = optimal_model{8};
            %optimal_ir = optimal_model{9};
            %optimal_elapsed_time = optimal_model{10};
            %optimal_store_GAMMA = optimal_model{11};
            %optimal_store_ALPHA = optimal_model{12};

            %clear model;
            %if length(optimal_model)==13
            %    optimal_betas = optimal_model{13};
            %    model.betas = optimal_betas;
            %end


            %model.A = optimal_A;%A_est;
            %model.lambdas = optimal_lambdas;%lambdas_est;
            %model.lambdas_tt = optimal_lambdas_tt;
            %model.s = optimal_PAI;%PAI_est;
            %model.P = optimal_PM; %P_est;
            %model.Qest = optimal_Qest;%Qest;
            %model.B = optimal_B; %B_est;
            %model.store_GAMMA = optimal_store_GAMMA;
            %model.store_ALPHA = optimal_store_ALPHA;
            %model.total_ll = optimal_total_ll;
            %model.n_it = optimal_ir;
            %model.elapsed_time = optimal_elapsed_time;
            %model.n_states = optimal_state;
            %model.optimal_d = optimal_d;
        end
    end

end