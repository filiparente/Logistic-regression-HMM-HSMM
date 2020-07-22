function models = run_hsmm(output_dir, varargin)
    prsr = inputParser;

    validstr = @(x) isstr(x);
    addRequired(prsr, 'output_dir', validstr);
    addParameter(prsr,'trainedmodel_path', false);
    addParameter(prsr,'dataset_path', 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\dt\1.0\new_cut_dataset.mat');
    addParameter(prsr, 'length_dur', 10);
    addParameter(prsr, 'geometric', false); %if geometric is false, P is a state duration matrix, if geometric is true, P is a geometric distribution where each state has one parameter
    addParameter(prsr, 'geometric_mode', 'normal'); %if geometric is false, P is a state duration matrix, if geometric is true, P is a geometric distribution where each state has one parameter
    addParameter(prsr, 'percentages', [0.8, 0.1, 0.1]);
    addParameter(prsr, 'n_montecarlo', 10);
    addParameter(prsr, 'normalize', false);
    addParameter(prsr, 'map', true);
    addParameter(prsr, 'pruning', false);
    addParameter(prsr, 'tuning', false);
    addParameter(prsr, 'kmax_hsmm', 0);
    addParameter(prsr, 'kmin_hsmm', 0);
    addParameter(prsr, 'kmax_hmm', 0);
    addParameter(prsr, 'kmin_hmm', 0);
    addParameter(prsr, 'optimal_d_hsmm', 0);
    addParameter(prsr, 'optimal_PM_est', 0);
    addParameter(prsr, 'pnmax', 1);
    addParameter(prsr, 'pnmin', 1e-4);
    addParameter(prsr, 'mode', false); %if true, all features are used to estimate the weights of the
    % logistic regression; if false, only the features of the timestamps
    % associated with a transition, according to the state sequence
    % estimated by the EM algorithm.
    %addParameter(prsr, 'n_montecarlo', 10);
    addParameter(prsr, 'init', 'smart'); %Poisson lambdas initialization: 'smart' or 'rand'
    addParameter(prsr, 'n_examples', 0);
    
    parse(prsr,output_dir, varargin{:});

    rng(1); %fix seed for reproducibility

    %Define the number of states of the markov chain
    geometric = prsr.Results.geometric;
    geometric_mode = prsr.Results.geometric_mode;
    map = prsr.Results.map;
    length_dur = prsr.Results.length_dur;
    n_montecarlo = prsr.Results.n_montecarlo;
    
    dataset_path = prsr.Results.dataset_path;
    trainedmodel_path = prsr.Results.trainedmodel_path;
    percentages = prsr.Results.percentages;
    
    pruning = prsr.Results.pruning;
    tuning = prsr.Results.tuning;
    kmax_hsmm = prsr.Results.kmax_hsmm;
    kmin_hsmm = prsr.Results.kmin_hsmm;
    kmax_hmm = prsr.Results.kmax_hmm;
    kmin_hmm = prsr.Results.kmin_hmm;
    optimal_d_hsmm = prsr.Results.optimal_d_hsmm;
    optimal_PM_est = prsr.Results.optimal_PM_est;
    pnmin = prsr.Results.pnmin;
    pnmax = prsr.Results.pnmax;
    mode = prsr.Results.mode;
    init = prsr.Results.init;
    n_montecarlo2 = 50;
%     if ~pruning
%         kmax_hsmm=n_states;
%         kmin_hsmm=n_states;
%         kmax_hmm=n_states;
%         kmin_hmm=n_states;
%     end

    files = dir(dataset_path);

    idxs = find(dataset_path=='\');
    
    if length(files)>1
        %TF-IDF features: alphabetically order dev-test-train
        dev_data = load([dataset_path(1:idxs(end)), files(1).name]);
        test_data = load([dataset_path(1:idxs(end)), files(2).name]);
        train_data = load([dataset_path(1:idxs(end)), files(3).name]);
        
        dev_data.y = double(dev_data.y);%load([dataset_path(1:idxs(end)), files(1).name]);
        test_data.y = double(test_data.y);%load([dataset_path(1:idxs(end)), files(2).name]);
        train_data.y = double(train_data.y);%load([dataset_path(1:idxs(end)), files(3).name]);
        
        dev_data.X = double(dev_data.X);%load([dataset_path(1:idxs(end)), files(1).name]);
        test_data.X = double(test_data.X);%load([dataset_path(1:idxs(end)), files(2).name]);
        train_data.X = double(train_data.X);%load([dataset_path(1:idxs(end)), files(3).name]);
        
        
        start_date = double(train_data.start_date);
        end_date = double(test_data.end_date);
        
        data.y = double([train_data.y, dev_data.y, test_data.y]);
        data.X = double([train_data.X; dev_data.X; test_data.X]);
        
        disc_unit = double(train_data.disc_unit); %discretization unit in hours
        window_size = double(train_data.window_size); %length of the window to average the tweets
        
        train_dev_test_split = false; %already done!
    else
        %BERT features
        
        data = load([dataset_path(1:idxs(end)), files.name]);
        %%FOR DEBUGGING ONLY
        %%if ~exist('json_data','var')
        %    %fname = 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\dt\1.0\new_cut_dataset.txt'; %new_small_dataset.txt'; %3.1\dataset.txt';%'C:\Users\Filipa\Desktop\Predtweet\results\3.1\dataset.txt'; 
        %    %json_data = loadjson(fname);
        %    data = load(dataset_path);
        %    data = data.r;
        %%end
        
        start_date = double(data.start_date);
        end_date = double(data.end_date);
        
        disc_unit = double(data.disc_unit); %discretization unit in hours
        window_size = double(data.window_size); %length of the window to average the tweets

        train_dev_test_split = true;
    end
        
    
    wholeSecs = floor(double(start_date)/1e9);
    fracSecs = double(start_date - uint64(wholeSecs)*1e9)/1e9;
    pd_start_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);

    
    wholeSecs = floor(double(end_date)/1e9);
    fracSecs = double(end_date - uint64(wholeSecs)*1e9)/1e9;
    pd_end_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);
    disp(['Start date: ' datestr(pd_start_date) ' and end date: ' datestr(pd_end_date)])

    %embeddings = cell2mat(data.embeddings);

    obs_seq = double(data.y'); %vertcat(embeddings.y); %counts
    X = data.X'; %cell2mat(vertcat(embeddings.X))'; %embeddings.X'; %cell2mat(vertcat(embeddings.X))'; %features

    disp(['Observation sequence stats: min ' num2str(min(obs_seq)) ' max ' num2str(max(obs_seq)) ' mean ' num2str(mean(obs_seq)) ' std ' num2str(std(obs_seq))])

    figure
    plot(obs_seq);

    %Cut initial small counts
    if prsr.Results.n_examples
        if length(obs_seq)>=prsr.Results.n_examples
            %Find first jump
            %idx = find((diff(obs_seq)./obs_seq(2:end))*100>90);
            
            idx = find(obs_seq>1000);
            
            diff_ = diff(obs_seq);
            diff_sorted = sort(diff_);
            values_diff_ = diff_sorted(end-9:end);
            idx2=inf;
            for i=1:10
                [idx_,~] = find(values_diff_(i)==diff_);
                
                idx_
                if idx_<idx2
                    idx2 = idx_;
                end
            end
            %cut observation sequence and features (drop first samples that correspond
            %to small tweet counts
            obs_seq = obs_seq(idx2+1:end); %em vez de idx(1)
            X = X(:, idx2+1:end); %em vez de idx(1)
        end
    end
            
            

    hold on
    plot(obs_seq);
    %drawnow;

    all_obs_seq = obs_seq;
    all_X = X;

    if train_dev_test_split
        % Use train/dev/testtest split 80%10%10% to split our data into train and validation sets for training
        len_dataset = length(all_obs_seq);

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

        train_obs_seq = all_obs_seq(1:lengths(1));
        train_X = all_X(:, 1:lengths(1));
        dev_obs_seq = all_obs_seq(lengths(1)+window_size+1:lengths(2));
        dev_X = all_X(:, lengths(1)+window_size+1:lengths(2));
        test_obs_seq = all_obs_seq(lengths(2)+window_size+1:end);
        test_X = all_X(:, lengths(2)+window_size+1:end);
    else
        [~,c] = find(train_data.y==obs_seq(1));
        train_obs_seq = train_data.y(c:end)';
        train_X = full(train_data.X(c:end,:)');
        dev_obs_seq = dev_data.y';
        dev_X = full(dev_data.X');
        test_obs_seq = test_data.y';
        test_X = full(test_data.X');
    end
        
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
            
            %min max scaling
            %maxV = max(train_X(feature,:));
            %minV = min(train_X(feature,:));
            %train_X(feature,:)   = (train_X(feature,:) - minV) / (maxV - minV);
            
            dev_X(feature,:)=(dev_X(feature,:)-train_mean(feature))/train_std(feature);
            test_X(feature,:)=(test_X(feature,:)-train_mean(feature))/train_std(feature);
        end
    end

    %obs_seq = vertcat(json_data{:,2});

    states = [];
    or_transf_A = [];
    or_lambdas = [];

    K = length(unique(obs_seq));
    % if N~=1
    %     x=cell2mat(obs_seq)';
    %     s=cell2mat(state_seq)';
    % else
    %     x=obs_seq';
    %     s=state_seq';
        % end
        
    if ~trainedmodel_path
        models = logit_poissonHsmmEm_new(train_obs_seq, train_X, dev_X, dev_obs_seq, map, optimal_d_hsmm, optimal_PM_est, geometric, geometric_mode, K, size(train_obs_seq,1), size(train_obs_seq,2), or_transf_A, or_lambdas, prsr.Results.n_montecarlo, kmin_hmm, kmax_hmm, kmin_hsmm, kmax_hsmm, pnmin, pnmax, mode, init, output_dir);
       %logit_poissonHsmmEm(train_obs_seq, train_X, n_states, states, K, length_dur, size(train_obs_seq,1), size(train_obs_seq,2), or_transf_A, or_lambdas, prsr.Results.compare);

        models2 = models;
        clear models;
        models.hsmmLr_model = models2{1};
        models.hsmm_model = models2{2};
        models.hmm_model = models2{3};
        %models.hmmLr_model = models2{4};
        
        save(strcat(output_dir, 'models.mat'), 'models');
    else
        %Load trained model
        models = load(trainedmodel_path);
        models = models.models;
        
        models2 = cell(1,1);%3);
        models2{1} = models.hsmmLr_model;
        %models2{2} = models.hsmm_model;
        %models2{3} = models.hmm_model;
    end
    
    for j=1:length(models2) %1:2 %1:length(models)
        
        if j==1
            output_dir_ = [output_dir 'hsmmLR\'];
            model_str = 'HSMM-LR';
            model = models.hsmmLr_model; %models.results.hsmmLr_model
        elseif j==2
            output_dir_ = [output_dir 'hsmm\'];
            model_str = 'HSMM';
            model = models.hsmm_model; %models.results.hsmm_model
        elseif j==3
            output_dir_ = [output_dir 'hmm\'];
            model_str = 'HMM';
            model = models.hmm_model; %models.results.hsmm_model
        end
        
        if ~exist(output_dir_, 'dir')
               mkdir(output_dir_)
        end
            
        %results.hsmmLr_model = models{1};
        %results.hsmm_model = models{2};
        train_est_state_seq = model.Qest;

        % all combinations of predicted lambdas, pairwise
        %combos = nchoosek(model.lambdas,2);

        %for i=1:size(combos,1)
        %    % euclidean distance between each pair of lambdas
        %    edist(i) = diff(combos(i,:)).^2;
        %end

        % find the minimum value
        %min_ = min(edist);

        % divide it by the average ????
        %avg_ = mean(edist);

        %min_ = min_/avg_;

        %distance between the observation sequence and the predictions
        %tmp = (obs_seq - est_state_seq).^2;
        %error = sum(tmp(:))/numel(obs_seq);

        % point (x,y) = (distance, error)
        %points_x(montecarlo) = min_;
        %points_y(montecarlo) = error;

        %% TRAIN RESULTS
        print_seq(output_dir_, train_obs_seq, model.lambdas(train_est_state_seq), 'train', model_str)
            
        disp('MSE OBSERVATIONS (TRAIN)');
        tmp = (model.lambdas(train_est_state_seq)' - train_obs_seq).^2;

        mse_train = sum(tmp(:))/numel(train_obs_seq)
        results.mse_train = mse_train;
        
        %% DEV RESULTS            
        %[mse, prev_alpha, predicted_count, out_A] = evaluate(model, model.A(:,:,end), dev_X, dev_obs_seq, map, train_obs_seq, false);
        best_mse_dev = inf;
        for z=1:n_montecarlo2
            if j<3   
                [mse_dev(z), ~, predictions, ~] = evaluate_hsmm(model, model.A(:,:,end), dev_X, dev_obs_seq, map, train_obs_seq(end), false, false, true); 
            else
                [mse_dev(z), ~, predictions, ~] = evaluate_hmm(model, dev_obs_seq, map, false, true);     
            end
       
            if mse_dev(z) < best_mse_dev
                clear predicted_count;
                predicted_count = predictions;
            end
        end
        
        results.dev_est_obs_seq = predicted_count;

        disp('MSE OBSERVATIONS (DEV)');          
        results.mean_mse_dev = mean(mse_dev)
        results.std_mse_dev = std(mse_dev)
        results.n_montecarlo = n_montecarlo;
        
        
        %results.dev_est_obs_seq = model.pred_val;%predicted_count;
        %results.mse_mean_dev = model.mean_pred_errors_val %mse;
        %results.mse_std_dev = model.std_pred_errors_val %mse;

        
        %print_seq(output_dir_, dev_obs_seq, predicted_count, 'dev')
        print_seq(output_dir_, dev_obs_seq, model.pred_val, 'dev', model_str)

        %clear predicted_count;

             
        %% TEST RESULTS
        best_mse_test = inf;
        for z=1:n_montecarlo2
            if j<3
                [mse_test(z), ~, predictions, ~] = evaluate_hsmm(model, model.out_A, test_X, test_obs_seq, map, dev_obs_seq(end), model.prev_state_posteriors, false, true); 
            else
                [mse_test(z), ~, predictions, ~] = evaluate_hmm(model, test_obs_seq, map, model.prev_state_posteriors, true);
            end
       
            if mse_test(z) < best_mse_test
                clear predicted_count;
                predicted_count = predictions;
            end
        end
        
        results.test_est_obs_seq = predicted_count;

        disp('MSE OBSERVATIONS (TEST)');          
        results.mean_mse_test = mean(mse_test)
        results.std_mse_test = std(mse_test)

        print_seq(output_dir_, test_obs_seq, predicted_count, 'test', model_str)

        clear predicted_count;

        save([output_dir_ 'results.mat'],'results');
            
        %All together
        print_seq(output_dir_, all_obs_seq, [model.lambdas(train_est_state_seq'), results.dev_est_obs_seq, results.test_est_obs_seq], 'all', model_str)

        %tot_points_x(i) = mean(points_x);
        %tot_points_y(i) = mean(points_y);

        %clear points_x;
        %clear points_y;
    end


end

function print_seq(output_dir_, obs_seq, predicted_count, mode,model)
    %% Prints
    FigH = figure('Position', get(0, 'Screensize'),'visible','on');
    scatter(1:1:length(obs_seq), obs_seq, 50, 'c', '.')
    hold on
    plot(obs_seq, 'c', 'LineWidth', 1) %test_obs_seq
    hold on
    %plot(model.lambdas(test_est_state_seq), 'm')
    scatter(1:1:length(predicted_count), predicted_count, 50, 'm', '.')
    hold on
    plot(predicted_count,'m', 'LineWidth', 1)
    hold on
    lgd = legend({['True ' mode ' observation sequence ' model],'', [' Estimated ' mode ' observation sequence ' model],''});
    lgd.Location = 'northeast';
    ylabel('State')
    xlabel('t')
    %drawnow
    
    F=getframe(FigH);
    
    if output_dir_
        saveas(gcf,[output_dir_ '_' mode '_obs_preds_seq.png']);
    end

end