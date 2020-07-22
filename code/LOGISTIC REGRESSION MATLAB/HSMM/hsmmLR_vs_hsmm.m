function results = hsmmLR_vs_hsmm(output_dir, varargin)
    prsr = inputParser;

    validstr = @(x) isstr(x);
    addRequired(prsr, 'output_dir', validstr);
    addParameter(prsr,'n_states',3);
    addParameter(prsr,'n_features',3);
    addParameter(prsr,'n_X_classes',-1);
    addParameter(prsr, 'length_dur', 10);
    addParameter(prsr, 'max_obs', 20);
    addParameter(prsr, 'original_lambdas', [8,9,12]); 
    addParameter(prsr, 'N', 1);
    addParameter(prsr, 'dim', 1000); 
    addParameter(prsr, 'mu', [0 2 0; 0 -2 0; 2 0 0; 0 0 2; 0 0 -2; -2 0 0]);
    addParameter(prsr, 'sigma', [0.2 0.05 0.05]);
    addParameter(prsr, 'legend', true);
    addParameter(prsr, 'betas', [ 0 0 0 0; 0 1 0 0; 0 -1 0 0; 1 0 0 0; 0 0 0 0; -1 0 0 0;0 0 -1 0; 0 0 1 0; 0 0 0 0]');
    addParameter(prsr, 'percentages', [0.8 0.1 0.1]);
    addParameter(prsr, 'normalize', true);
    addParameter(prsr, 'map', true);
    addParameter(prsr, 'transition_matrix', 0);
    
    %validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
    %addRequired(p,'width',validScalarPosNum);
    %addOptional(p,'height',defaultHeight,validScalarPosNum);
    %addParameter(p,'units',defaultUnits,@isstring);
    %addParameter(p,'shape',defaultShape,...
    %              @(x) any(validatestring(x,expectedShapes)));
    parse(prsr,output_dir, varargin{:});

    rng(1); %fix seed for reproducibility

    %Define the number of states of the markov chain
    n_states = prsr.Results.n_states;

    %Define the number of features per state
    n_features = prsr.Results.n_features;
    
    map = prsr.Results.map;

    %Number of feature classes
    n_X_classes = prsr.Results.n_X_classes;
    if n_X_classes == -1
       n_X_classes = n_states^2-n_states;
    end

    %Define the maximum length of state duration
    length_dur = prsr.Results.length_dur;

    %Max observable value
    max_obs = prsr.Results.max_obs;
    
    %Train/dev/test split percentages
    percentages = prsr.Results.percentages;

    %Create the markov modulated poisson process original model
    original_lambdas = prsr.Results.original_lambdas;

    % Initialize matrices
    original_PAI = generate_random_initial_dist(n_states);%ones(n_states,1)*(1/n_states);
    original_A = generate_random_transition_matrix_d(n_states); %[0,    1/(n_states-1),    1/(n_states-1);1/(n_states-1),    0,    1/(n_states-1);1/(n_states-1),    1/(n_states-1),    0];
    %original_P = zeros(n_states, length_dur);
    %original_P(:,end) = 1;
    original_P = generate_random_dur_dist(n_states, length_dur);%[0 0 1 0 0; 0 1 0 0 0; 0 0 0 0 1];%generate_random_dur_dist(n_states, length_dur);
    original_B = generate_random_dur_dist(n_states, max_obs);

    %Number of observations
    N = prsr.Results.N;

    %Dimension of the observations
    dim = prsr.Results.dim;

    % GENERATE FEATURE DATA X
    X = zeros(n_features, dim);

    % Parameters of the normal distributions (mean+std), one for each state/class
    mu = prsr.Results.mu;
    sigma = prsr.Results.sigma;
   
    %Weights of the logistic regression
    betas = prsr.Results.betas;

    % Set up a Markov model to simulate the transitions between the feature
    % vector X, for each class    
    transition_matrix = prsr.Results.transition_matrix;
    
    if ~transition_matrix
        transition_matrix = diag(ones(n_X_classes,1));
        factor = 0.1;
        for i=1:n_X_classes
           for j=1:n_X_classes
               if i~=j
                   transition_matrix(i,j) = transition_matrix(i,j)+factor/n_X_classes/(n_X_classes-1);
               else
                   transition_matrix(i,j) = transition_matrix(i,j)-factor/n_X_classes;
               end
           end
        end
    end

    assert(all(sum(transition_matrix,2)),'The transition matrix for the states of the features X does not sum to 1 along the rows.')

    %we don't care about the emissions, only the transitions
    emission_matrix = zeros(n_X_classes);

    %[~,states] = hmmgenerate(dim, transition_matrix, emission_matrix); %no pi, hmmgenerate always starts at state 1
    model.A = transition_matrix;
    model.PAI = zeros(n_states,1);
    model.PAI(1)=1;
    model.P = zeros(n_X_classes,0.05*dim);
    model.P(:,end)=1;
    states = markovMySample(model, dim, 1);
   
    clear model;
    
    % Using the state sequence for X, sample from the gaussian distribution
    % of the corresponding state
    for i=1:n_X_classes
       X(:, states==i) = mvnrnd(mu(i,:),sigma,length(X(states==i)))';
    end

    X=X';
    
    % Plot features
    FigH = figure('Position', get(0, 'Screensize'));

    colors = {[0 0 1], [1 0 0], [0 1 0], [0 1 1], [1 128/255 0], [1 0 1]};
    for j=1:n_X_classes
       if n_features==3
           %IN 3D:
           %scatter3(X(:,1), X(:,2), X(:,3),20,'.')
           scatter3(X(states == j,1), X(states == j,2), X(states == j,3), 50, colors{j}, '.');
       elseif n_features==2
           %IN 2D:
           %scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
           scatter(X(states == j,1), X(states == j,2), 50, colors{j}, '.');
       end
       hold on
    end
    if prsr.Results.legend && n_features==3
        legend(['State transition 1->2', %Features associated with state transition...
               'State transition 1->3',
               'State transition 2->1',
               'State transition 3->2',
               'State transition 3->1',
               'State transition 2->3'])
    elseif prsr.Results.legend && n_features==2
        legend(['State transition *->1', %Features associated with state transition...
               'State transition *->2',
               'State transition *->3'])
    end
           
    X=X';
    
    if n_features==3
        [Xs,Ys,Zs] = sphere; %unit sphere centered around the origin
        r = sqrt(sum(sigma)); %root mean squared
        X2 = Xs * r;
        Y2 = Ys * r;
        Z2 = Zs * r;
        for j=1:n_X_classes
            h = surf(X2+mu(j,1),Y2+mu(j,2),Z2+mu(j,3));%, 1, 'FaceColor', 'none','EdgeColor', colors{j});
            set(h,'FaceColor',colors{j}, 'FaceAlpha',0.05,'FaceLighting','gouraud','EdgeColor','none')
            daspect([1 1 1]);
            hold on;
        end
        grid on;
        hold on;
        scatter3(mu(:,1), mu(:,2), mu(:,3), 200, 'k', '+')
        zlabel('Third feature')
    elseif n_features==2
        scatter(mu(:,1), mu(:,2), 200, 'k', '+')
    end
    xlabel('First feature')
    ylabel('Second feature')
    
    F = getframe(FigH);

    saveas(gcf,strcat(output_dir,'features.png'));
    saveas(gcf,strcat(output_dir,'features.fig'));
    
    
    % Sample first state from pi
    %state = sampleDiscrete(original_PAI);
    %state_seq = zeros(dim,N);
    %obs_seq = zeros(dim,N);

    %p = zeros(n_states, 1);
    %store_p = zeros(dim, n_states);

    or_transf_A = zeros(n_states, n_states, dim);
    %logit_model.W = betas;
    %[~, P] = logitMnPred(logit_model, X); %y is the predicted state sequence and P is the state posteriors according to the logistic regression model

    %or_transf_A = permute(reshape(P, n_states, n_states, dim), [2 1 3]);
    for t=1:dim
        for i=1:n_states
            for j=1:n_states
                if i~=j
                    column = get_columns_(n_states,i,j);
                    columns = get_columns_(n_states, i);
                    columns = columns(columns~=get_columns_(n_states,i,i));
                    or_transf_A(i,j,t) = exp(betas(:,column)'*[X(:,t);1])/sum(exp(betas(:,columns)'*[X(:,t);1]));
                else
                    or_transf_A(i,j,t) = 0.0;
                end 
            end
        end
    end
    or_transf_A = normalize(or_transf_A,2);
    states=states';
    [Vk, obs_seq, state_seq] = hsmmSample_(original_PAI,or_transf_A,original_P,original_B, original_lambdas, dim,N);

    K = sum(unique(Vk));
    
    % Use train/dev/testtest split 80%10%10% to split our data into train and validation sets for training
    len_dataset = length(obs_seq);

    lengths = uint64(ceil(len_dataset*percentages));
    diff_ = uint64(sum(lengths)-len_dataset);
    if diff_>0
        %subtract 1 starting from the end
        for i=length(lengths)-1:-1:-1
            lengths(i) = lengths(i)-1;
            diff_=diff_-1;
            if diff_==0
               break
            end
        end
    end

    lengths = cumsum(lengths);

    assert(length(unique(states(1:lengths(1))))==n_X_classes, 'Not all state transitions are reflected in the features X');
    
    
    train_obs_seq = obs_seq(1:lengths(1));
    train_state_seq = state_seq(1:lengths(1));
    train_X = X(:, 1:lengths(1));
    train_or_transf_A = or_transf_A(:,:,1:lengths(1));
    dev_obs_seq = obs_seq(lengths(1)+1:lengths(2))';
    dev_X = X(:, lengths(1)+1:lengths(2));
    dev_or_transf_A = or_transf_A(:,:,lengths(1)+1:lengths(2));
    test_obs_seq = obs_seq(lengths(2)+1:end)';
    test_X = X(:, lengths(2)+1:end);
    test_or_transf_A = or_transf_A(:,:,lengths(2)+1:end);

    disp(['Number of points in train dataset = ', num2str(length(train_obs_seq))]);
    disp(['Number of points in dev dataset = ', num2str(length(dev_obs_seq))]);
    disp(['Number of points in test dataset = ', num2str(length(test_obs_seq))]);

    if prsr.Results.normalize
        % Normalization (z-score)
        for feature=1:size(train_X,1) %for all features, normalize independently for each feature
            % Get the z-score parameters from the training set (mean and std) 
            train_mean(feature) = mean(train_X(feature,:));
            train_std(feature) = std(train_X(feature,:));

            % Z-score the whole dataset with the parameters from training
            % z=(x-mean)/std
            train_X(feature,:)=(train_X(feature,:)-train_mean(feature))/train_std(feature);
            dev_X(feature,:)=(dev_X(feature,:)-train_mean(feature))/train_std(feature);
            test_X(feature,:)=(test_X(feature,:)-train_mean(feature))/train_std(feature);
        end
    end

    if N~=1
        x=cell2mat(train_obs_seq)';
        s=cell2mat(train_state_seq)';
    else
        x=train_obs_seq';
        s=train_state_seq';
    end
    
    
    compare = true;
    models = logit_poissonHsmmEm_new(x, train_X, n_states, states, K, length_dur, size(train_obs_seq,2), N, train_or_transf_A, original_lambdas, compare, n_states, n_states, 1, 1);

    mean_or_transf_A = mean(train_or_transf_A, 3);

    for i=1:length(models)
        model = models{i};
        
        if i==1
            output_dir_ = [output_dir '\hsmmLr'];
        else
            output_dir_ = [output_dir '\hsmm'];
        end
        %MUNKRES Find the optimal assginment of states
        [assignment, total_cost] = munkres2(n_states, n_states, s, model.Qest);

        %Compute final estimated state sequence according to the assignment
        Qest_final = zeros(size(model.Qest));
        for j=1:length(assignment)
            Qest_final(model.Qest==j) = assignment(j);
        end

        %Compute final parameters according to the assignment
        %With the indexes association, permute the rows and columns of the predicted 
        %transition matrix so that it matches the true transition matrix
        %Same for the duration probability density function (but only rows, since rows are states and columns are durations)
        B_est_final = zeros(size(model.B));
        P_est_final = zeros([n_states, length_dur]);
        %PM_est_final = zeros(1,n_states);
        PAI_est_final = zeros(n_states,1);
        lambdas_est_final = zeros(1, n_states);
        prev_assign = [];
        for j=1:length(assignment)
            P_est_final(assignment(j),:) = model.P(j,:);
            %PM_est_final(assignment(j))= model.P(j);
            B_est_final(assignment(j),:) = model.B(j,:);
            PAI_est_final(assignment(j)) = model.s(j);
            lambdas_est_final(assignment(j)) = model.lambdas(j);
            [v,idx] = find(j==prev_assign(:));

            if (~isempty(v) && assignment(j) == v) || j==assignment(j) %the swap was already performed
                prev_assign = [prev_assign, assignment(j)];
                continue;
            else

                model.A([j, assignment(j)], :,:) = model.A([assignment(j),j], :,:);     % swap rows.
                model.A(:, [j, assignment(j)],:) = model.A(:, [assignment(j),j],:);     % swap columns.
                prev_assign = [prev_assign, assignment(j)];
            end 
        end

        model.B = B_est_final;
        model.P = P_est_final;
        model.PAI = PAI_est_final;
        model.Qest = Qest_final;
        model.lambdas = lambdas_est_final;
        
        results.model = model;

        %MSE OBSERVATIONS
        model_Qest = model.Qest;
        model_lambdas = model.lambdas';
        disp('MSE OBSERVATIONS');
        tmp = (model_lambdas(model_Qest) - x).^2;
        mse_train = sum(tmp(:))/numel(x);
        results.mse_train = mse_train;

        %MSE MATRIZ DE TRANSIÇÃO At (for each t)
        size_A = size(model.A);
        if size_A(end)>n_states %At
            disp('MSE MATRIZ DE TRANSIÇÃO At');
            tmp = (train_or_transf_A - model.A).^2;
            mse_At = sum(tmp(:))/numel(train_or_transf_A);
            results.mse_At = mse_At;
            
            %MSE LOG ODDS REAL BETAS
            %AGORA OS LOG ODDS NÃO SÃO IGUAIS PORQUE SÃO CALCULADOS NÃO TENDO EM CONTA
            %QUE A DIAGONAL DA MATRIZ DE TRANSIÇÃO É NULA
            %real_log_odds = [];
            %k=n_states;
            %for j=1:k
            %        logit_model.W = betas(:,j*k-1:j*k);
            %        [~, log_odds] = logitMnPred(logit_model, X);
            %        real_log_odds = [real_log_odds; log_odds];
            %end
            %logit_model.W = betas;
            %[~, log_odds] = logitMnPred(logit_model, X); %y is the predicted state sequence and P is the state posteriors according to the logistic regression model

            %disp('MSE LOG ODDS');
            %mse_logodds = mse(real_log_odds, model.log_odds)
            %results.mse_logodds = mse_logodds;
        end

        %MSE mean At or A
        mean_model_A = mean(model.A,3);
        disp('MSE MATRIZ DE TRANSIÇÃO A');
        tmp = (mean_or_transf_A - mean_model_A).^2;
        mse_meanA = sum(tmp(:))
        results.mse_meanA = mse_meanA;

        %MSE PARAMETERS (LAMBDAS)
        disp('MSE LAMBDAS');
        tmp = (model.lambdas - original_lambdas).^2;
        mse_lambdas = sum(tmp(:))
        results.mse_lambdas = mse_lambdas;

        %est_state_seq = hmmViterbi(obs_seq', model);

        train_est_state_seq = model.Qest;
        results.train_est_state_seq = train_est_state_seq;
        results.train_obs_state_seq = model.lambdas(train_est_state_seq);
        
        %delay = 0;
        
        FigH = figure('Position', get(0, 'Screensize'), 'visible','on');
        subplot(3,1,1);
        plot(states(1:lengths(1)), 'g')
        lgd = legend('True state sequence X');
        lgd.Location = 'northeast';
        ylim([0.9 n_X_classes+0.3])
        ylabel('State')
        %title(['Delay = ', num2str(delay), ' samples'])
        subplot(3,1,2);
        plot(train_obs_seq, 'b') 
        hold on
        plot(train_est_state_seq, 'r')
        hold on
        ylim([0.9 n_states+.3]);
        ylabel('State')
        lgd = legend({'True state sequence HSMM', 'Estimated state sequence HSMM'});
        lgd.Location = 'northeast';
        subplot(3,1,3); 
        plot(x, 'c')
        hold on
        plot(model.lambdas(train_est_state_seq), 'm')
        hold on
        lgd = legend({'True observation sequence HSMM', 'Estimated observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('Observation')
        xlabel('t')
        
        F=getframe(FigH);
        saveas(gcf,[output_dir_  '_train.png']);

        
        %% DEV RESULTS
        %Predict future observations
        %initial_state = train_est_state_seq(end); %TODO: check if the first state in Qest is 0 or 1
        %ss = zeros(n_states,1);
        %ss(initial_state)=1;
        
        state_posteriors = model.store_GAMMA(:,end);
        last_state_dur = squeeze(model.store_ALPHA(:,:,end));
        
        size_A = size(model.A); 
        if size_A(end)>n_states %HSMM_LR
            %Get transition matrix from features
            new_A_dev = zeros(n_states, n_states, size(dev_obs_seq,1));
            for g=1:n_states
                for k=setdiff(1:n_states,g) 
                    %column = get_columns_(M,i,j);
                    for t=1:size(dev_obs_seq,1)
                        new_A_dev(g,k,t) = exp(model.betas(g,:,k)*[dev_X(:,t);1]-logsumexp(reshape(model.betas(g,:,setdiff(1:n_states,g)), size(dev_X,1)+1, n_states-1)'*[dev_X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
                        %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
                    end
                end 
            end
            
            
            %[Vk, obs_seq, state_seq] = hsmmSample_(ss,new_A_dev,model.P,model.B, model.lambdas, size(dev_obs_seq,1),size(dev_obs_seq,2));
            %[predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, dev_or_transf_A, model.lambdas, dev_obs_seq, n_states, map, model.P);
            
            prev_alpha = squeeze(model.store_ALPHA(:,:,end));
                                                                 %dev_or_transf_A
            [predicted_count, next_alpha] = predict_future2(prev_alpha, cat(3,model.A(:,:,end), new_A_dev), model.lambdas, dev_obs_seq, n_states, map, model.P, original_P,train_obs_seq(end));
            
            prev_alpha = next_alpha;
        
        else
            prev_alpha = squeeze(model.store_ALPHA(:,:, end));
            
            %[Vk, obs_seq, state_seq] = hsmmSample(ss,model.A,model.P,model.B, model.lambdas, size(dev_obs_seq,1),size(dev_obs_seq,2));
            %[predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, model.A, model.lambdas, dev_obs_seq, n_states, map, model.P);
            [predicted_count, next_alpha] = predict_future2(prev_alpha, model.A, model.lambdas, dev_obs_seq, n_states, map, model.P, original_P, train_obs_seq(end));

            prev_alpha = next_alpha;
            
        end

        %dev_est_state_seq = state_seq;
        %results.dev_est_state_seq = dev_est_state_seq;
        results.dev_est_obs_seq = predicted_count;
            
        FigH = figure('Position', get(0, 'Screensize'), 'visible','on');
        plot(dev_obs_seq, 'c') %test_obs_seq
        hold on
        %plot(model.lambdas(dev_est_state_seq), 'm')
        plot(predicted_count, 'm')
        hold on
        lgd = legend({'True dev observation sequence HSMM', 'Estimated dev observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('State')
        xlabel('t')
        %drawnow

        F=getframe(FigH);
        saveas(gcf,[output_dir_  '_best_val_obs_preds_seq.png']);

        % dim=length(train_obs_seq); %test_obs_seq
      
        dim=length(dev_obs_seq);
        K = length(unique(dev_obs_seq)); %test_obs_seq
        [~,~,~,~,Vk,~,K, ~] = hsmmInitialize(dev_obs_seq,n_states,length_dur,K,dim*ones(N,1)); %test_obs_seq

        B = zeros(n_states, length(Vk));
        for j=1:n_states
              %ss(i)=sum(log(1:i));
              B(j,:) = poisspdf(Vk, model.lambdas(j))';
              B(j,:)=B(j,:)/sum(B(j,:));
        end        
        if i==1
            [~, lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,~,~,lkh, ll, ir]=hsmm_new_(model.lambdas, model.s, cat(3,model.A(:,:,end), new_A_dev), B,model.P,dev_obs_seq,0,length(dev_obs_seq)*ones(N,1), 1e-10, Vk, dev_X, [], [], model.betas); %test_obs_seq instead of all_obs_seq and test_X instead of all_X
        else
            [lambdas_est, lambdas_tt, PAI_est,A_est,B_est,P_est,Qest,~,~,lkh, ll, ir]=hsmm_new(model.lambdas, model.s, model.A, B,model.P,dev_obs_seq,0,length(dev_obs_seq)*ones(N,1), 1e-10, Vk); %test_obs_seq instead of all_obs_seq and test_X instead of all_X
        end
         
        est_state_seq=Qest;
        figure
        plot(dev_obs_seq, 'c') %test_obs_seq
        hold on
        plot(lambdas_est(est_state_seq), 'm')
        hold on
        lgd = legend({'True observation sequence HSMM', 'Estimated observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('State')
        xlabel('t')
        drawnow

        F=getframe(FigH);
        saveas(gcf,[output_dir_  '_best_val_obs_preds_seq_viterbi.png']);
    

        disp('MSE OBSERVATIONS (DEV)');
        %tmp = (model.lambdas(dev_est_state_seq)' - dev_obs_seq).^2;
        tmp = (predicted_count'-dev_obs_seq).^2;
        mse_dev = sum(tmp(:))/numel(dev_obs_seq);

        results.mse_dev = mse_dev;
        %results.dev_est_obs_seq = model.lambdas(dev_est_state_seq);
        clear predicted_count;
        %% TEST RESULTS
        %Predict future observations
        %initial_state = dev_est_state_seq(end); %TODO: check if the first state in Qest is 0 or 1
        %ss = zeros(n_states,1);
        %ss(initial_state)=1;

        %state_posteriors = model.store_GAMMA(:,end);
        %last_state_dur = squeeze(model.store_ALPHAT(:,end,:));
        
        if size_A(end)>n_states %HSMM_LR
            %Get transition matrix from features
            new_A_test = zeros(n_states, n_states, size(test_obs_seq,1));
            for g=1:n_states
                for k=setdiff(1:n_states,g) 
                    %column = get_columns_(M,i,j);
                    for t=1:size(test_obs_seq,1)
                        new_A_test(g,k,t) = exp(model.betas(g,:,k)*[test_X(:,t);1]-logsumexp(reshape(model.betas(g,:,setdiff(1:n_states,g)), size(test_X,1)+1, n_states-1)'*[test_X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
                        %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
                    end
                end 
            end
            %[Vk, obs_seq, state_seq] = hsmmSample_(ss,new_A_test,model.P,model.B, model.lambdas, size(test_obs_seq,1),size(test_obs_seq,2));
            %[predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, new_A_test, model.lambdas, test_obs_seq, n_states, map, model.P);
            [predicted_count, next_alpha] = predict_future2(prev_alpha, cat(3,new_A_dev(:,:,end), new_A_test), model.lambdas, test_obs_seq, n_states, map, model.P, original_P,dev_obs_seq(end));

        else
            %[Vk, obs_seq, state_seq] = hsmmSample(ss,model.A,model.P,model.B, model.lambdas, size(test_obs_seq,1),size(test_obs_seq,2));
            %[predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, model.A, model.lambdas, test_obs_seq, n_states, map, model.P);
            
            [predicted_count, next_alpha] = predict_future2(prev_alpha, model.A, model.lambdas, test_obs_seq, n_states, map, model.P, original_P, dev_obs_seq(end));

        end

        %test_est_state_seq = state_seq;
        %results.test_est_state_seq = test_est_state_seq;
        results.test_est_obs_seq = predicted_count;

        FigH = figure('Position', get(0, 'Screensize'),'visible','on');
        plot(test_obs_seq, 'c') %test_obs_seq
        hold on
        %plot(model.lambdas(test_est_state_seq), 'm')
        plot(predicted_count,'m')
        hold on
        lgd = legend({'True test observation sequence HSMM', 'Estimated test observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('State')
        xlabel('t')
        %drawnow

        F=getframe(FigH);
        saveas(gcf,[output_dir_  '_test_obs_preds_seq.png']);


        disp('MSE OBSERVATIONS (TEST)');
        %tmp = (model.lambdas(test_est_state_seq)' - test_obs_seq).^2;
         tmp = (predicted_count'-test_obs_seq).^2;
        mse_test = sum(tmp(:))/numel(test_obs_seq);

        results.mse_test = mse_test;
        
        %results.test_est_obs_seq = model.lambdas(test_est_state_seq);
        clear predicted_count;
        results.states = states;
        results.or_transf_A = or_transf_A;
        
        if i==1
            save([output_dir 'hsmmLr_results.mat'],'results');
        else
            save([output_dir  'hsmm_results.mat'],'results');
        end
        
      
        %All together
        FigH = figure('Position', get(0, 'Screensize'),'visible','on');
        plot(obs_seq, 'c') %test_obs_seq
        hold on
        %plot(model.lambdas([train_est_state_seq', dev_est_state_seq, test_est_state_seq]), 'm')
        plot([model.lambdas(train_est_state_seq'), results.dev_est_obs_seq, results.test_est_obs_seq], 'm')
        hold on
        lgd = legend({'True test observation sequence HSMM', 'Estimated test observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('State')
        xlabel('t')
        %drawnow

        F=getframe(FigH);
        saveas(gcf,[output_dir_ '_all_obs_preds_seq.png']);
        
        clear results;
    end

    % Plot features with the original and estimated decision boundaries
    % X=X';
    % 
    % figure
    % %scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
    % % Plot feature points from the first class
    % scatter(X(states == 1,1), X(states == 1,2), 50, 'b', '.')
    % % Plot feature points from the second class.
    % hold on;
    % scatter(X(states == 2,1), X(states == 2,2), 50, 'r', '.')
    % % Plot feature points from the first class
    % hold on;
    % scatter(X(states == 3,1), X(states == 3,2), 50, 'g', '.')
    % % Plot feature points from the second class.
    % hold on;
    % scatter(X(states == 4,1), X(states == 4,2), 50, 'c', '.')
    % 
    % %model.betas

    % betas = [betas(end,:);betas];
    % betas = betas(1:end-1,:);
    % %plot true decision boundary
    % hold on
    % plot_decision_boundary(betas(:,1:2));
    % hold on
    % plot_decision_boundary(betas(:,3:4));
    % hold on
    % %plot estimated decision boundary
    % plot_decision_boundary(model.betas(:,1:2));
    % hold on
    % %plot estimated decision boundary
    % plot_decision_boundary(model.betas(:,3:4));
    % hold on
    % legend('Feature points from first class (11)', 'Feature points from second class (12)', 'Feature points from third class (21)', 'Feature points from forth class (22)', 'True decision boundary 11=12', 'True decision boundary 21=22', 'Estimated decision boundary 11=12', 'Estimated decision boundary 21=22')
    % title(['Delay = ', num2str(delay), ' samples'])
    % 


end

