function normalize_errors()
    %1) Reportar 6 métricas: 
    %MSE normal, 
    %RMSE normal, 
    %RMSE normalizado pela média das observações y, 
    %RMSE normalizado por ymax-ymin,
    %RMSE normalizado pela std(y),
    %RMSE normalizado pela diferença dos quantiles 0.75 e 0.25 de y,
    %FFT

    %2) guardar numa estrutura BERT_runx_prediction_report.mat
    
    prompt = 'Save prediction report for which run (accepted input: integer number 1,2,...) ?';
    run = input(prompt);

    dts = [1]; %[1,3,4,6,12,24,48];
    dws = [0];
    n_exampless = [4959, 1239, 826, 413, 206, 103];%[4959, 1653, 1239, 826, 413, 206, 103];

    models = { 'hmm', 'hsmm', 'hsmmLR'};

    dataset_path = 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\server\';
    path='C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\tests\real\';
    out_results = [];
    create_containers = @(n)arrayfun(@(x)containers.Map(), 1:n, 'UniformOutput', false);
    out_results = create_containers(length(dts)*length(dws));
    result = containers.Map;
    n=1;

    for i=1:length(dts)
        n_examples = n_exampless(i);
        dt = dts(i)

        for j=1:length(dws)
            dw=dws(j)
            window_size=dw;
            n_examples = n_examples-1;

            results_path = [path, num2str(dt),'.', num2str(dw),'\results\BERT\'];
            dataset_path_ = [dataset_path, num2str(dt),'.', num2str(dw),'\dataset.mat'];

            %Load data
            data = load(dataset_path_);

            obs_seq = double(data.y'); %vertcat(embeddings.y); %counts

            disp(['Observation sequence stats: min ' num2str(min(obs_seq)) ' max ' num2str(max(obs_seq)) ' mean ' num2str(mean(obs_seq)) ' std ' num2str(std(obs_seq))])

            %Cut initial small counts
            if n_examples
                %if length(obs_seq)>n_examples
                    %Find first jump
                    %idx = find((diff(obs_seq)./obs_seq(2:end))*100>90);

                    %idx = find(obs_seq>1000);

                    diff_ = diff(obs_seq);
                    diff_sorted = sort(diff_);
                    values_diff_ = diff_sorted(end-9:end);
                    idx2=inf;
                    for l=1:10
                        [idx_,~] = find(values_diff_(l)==diff_);

                        if idx_<idx2
                            idx2 = idx_;
                        end
                    end
                    %cut observation sequence and features (drop first samples that correspond
                    %to small tweet counts
                    obs_seq = obs_seq(idx2+1:end); %em vez de idx(1)
                    %X = X(:, idx2+1:end); %em vez de idx(1)
                %end
            end

            all_obs_seq = obs_seq;

            percentages = [0.5,0.25,0.25];
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
            dev_obs_seq = all_obs_seq(lengths(1)+window_size+1:lengths(2));
            test_obs_seq = all_obs_seq(lengths(2)+window_size+1:end);

            disp(['Number of points in train dataset = ', num2str(length(train_obs_seq))]);
            disp(['Number of points in dev dataset = ', num2str(length(dev_obs_seq))]);
            disp(['Number of points in test dataset = ', num2str(length(test_obs_seq))]);



            for z=1:length(models)
                model = models{z}

                results = load([results_path, 'run' num2str(run), '\', model, '\results.mat']);
                models_ = load([results_path, 'run' num2str(run), '\models.mat']);
                
                dev_est_obs_seq = results.results.dev_est_obs_seq;
                test_est_obs_seq = results.results.test_est_obs_seq;
                if strcmp(model,'hmm')
                    train_est_obs_seq = models_.models.hmm_model.Qest;
                elseif strcmp(model,'hsmm')
                    train_est_obs_seq = models_.models.hsmm_model.Qest;
                elseif strcmp(model,'hsmmLR')
                   train_est_obs_seq = models_.models.hsmmLr_model.Qest;
                end
                    
                %mse
                mse_train = results.results.mse_train;
                mse_mean_dev = results.results.mean_mse_dev;
                mse_std_dev = results.results.std_mse_dev;
                mse_mean_test = results.results.mean_mse_test;
                mse_std_test = results.results.std_mse_test;

        
                %rmse
                rmse_train = sqrt(mse_train)
                rmse_mean_dev = sqrt(mse_mean_dev)
                rmse_std_dev = sqrt(mse_std_dev)
                rmse_mean_test = sqrt(mse_mean_test)
                rmse_std_test = sqrt(mse_std_test)

                %normalize by mean y
                mnrmse_train = rmse_train/mean(train_obs_seq)
                mnrmse_mean_dev = rmse_mean_dev/mean(dev_obs_seq)
                mnrmse_std_dev = rmse_std_dev/mean(dev_obs_seq)
                mnrmse_mean_test = rmse_mean_test/mean(test_obs_seq)
                mnrmse_std_test = rmse_std_test/mean(test_obs_seq)

                %normalize by max(y)-min(y)
                mmnrmse_train = rmse_train/(max(train_obs_seq)-min(train_obs_seq))
                mmnrmse_mean_dev = rmse_mean_dev/(max(dev_obs_seq)-min(dev_obs_seq))
                mmnrmse_std_dev = rmse_std_dev/(max(dev_obs_seq)-min(dev_obs_seq))
                mmnrmse_mean_test = rmse_mean_test/(max(test_obs_seq)-min(test_obs_seq))
                mmnrmse_std_test = rmse_std_test/(max(test_obs_seq)-min(test_obs_seq))
                
                %normalize by std(y)
                snrmse_train = rmse_train/std(train_obs_seq)
                snrmse_mean_dev = rmse_mean_dev/std(dev_obs_seq)
                snrmse_std_dev = rmse_std_dev/std(dev_obs_seq)
                snrmse_mean_test = rmse_mean_test/std(test_obs_seq)
                snrmse_std_test = rmse_std_test/std(test_obs_seq)
                
                %normalize by quantile(y,0.75)-quantile(y,0.25)
                qnrmse_train = rmse_train/(quantile(train_obs_seq,0.75)-quantile(train_obs_seq,0.25))
                qnrmse_mean_dev = rmse_mean_dev/(quantile(dev_obs_seq,0.75)-quantile(dev_obs_seq,0.25))
                qnrmse_std_dev = rmse_std_dev/(quantile(dev_obs_seq,0.75)-quantile(dev_obs_seq,0.25))
                qnrmse_mean_test = rmse_mean_test/(quantile(test_obs_seq,0.75)-quantile(test_obs_seq,0.25))
                qnrmse_std_test = rmse_std_test/(quantile(test_obs_seq,0.75)-quantile(test_obs_seq,0.25))
                
                %FFT dev
                %[fft_srmse_phase_dev, fft_srmse_ampl_dev] = fft_(dev_obs_seq, results.results.dev_est_obs_seq, false);
                
                %FFT test
                %[fft_srmse_phase_test, fft_srmse_ampl_test] = fft_(test_obs_seq, results.results.test_est_obs_seq, false);

                result('dt') = dt;
                result('dw') = dw;
                result('model') = model;
                %result('fft_mse_phase_dev') = fft_srmse_phase_dev;
                %result('fft_mse_ampl_dev') = fft_srmse_ampl_dev;
                %result('fft_mse_phase_test') = fft_srmse_phase_test;
                %result('fft_mse_ampl_test') = fft_srmse_ampl_test;
                
                %mae
                mae_train = calMAE(train_obs_seq(1:length(train_est_obs_seq)),train_est_obs_seq)
                aux = train_obs_seq(length(train_est_obs_seq)+1:end);
                mae_dev = calMAE([aux;dev_obs_seq(1:length(dev_est_obs_seq)-length(aux))],dev_est_obs_seq)           
                aux2 = dev_obs_seq(length(dev_est_obs_seq)-length(aux)+1:end);
                mae_test = calMAE([aux2;test_obs_seq(1:end)],test_est_obs_seq)
        
                %rmse
                rmae_train = sqrt(mae_train)
                rmae_dev = sqrt(mae_dev)
                rmae_test = sqrt(mae_test)
               
                %normalize by mean y
                mnrmae_train = rmae_train/mean(train_obs_seq)
                mnrmae_dev = rmae_dev/mean(dev_obs_seq)
                mnrmae_test = rmae_test/mean(test_obs_seq)

                %normalize by max(y)-min(y)
                mmnrmae_train = rmae_train/(max(train_obs_seq)-min(train_obs_seq))
                mmnrmae_dev = rmae_dev/(max(dev_obs_seq)-min(dev_obs_seq))
                mmnrmae_test = rmae_test/(max(test_obs_seq)-min(test_obs_seq))
                
                %normalize by std(y)
                snrmae_train = rmae_train/std(train_obs_seq)
                snrmae_dev = rmae_dev/std(dev_obs_seq)
                snrmae_test = rmae_test/std(test_obs_seq)
                
                %normalize by quantile(y,0.75)-quantile(y,0.25)
                qnrmae_train = rmae_train/(quantile(train_obs_seq,0.75)-quantile(train_obs_seq,0.25))
                qnrmae_dev = rmae_dev/(quantile(dev_obs_seq,0.75)-quantile(dev_obs_seq,0.25))
                qnrmae_test = rmae_test/(quantile(test_obs_seq,0.75)-quantile(test_obs_seq,0.25))             
         
                mae = containers.Map;
                mae('train') = mae_train;
                mae('dev') = mae_dev;
                mae('test') = mae_test;

                result('mae') = mae;
                
                rmae = containers.Map;
                rmae('train') = rmae_train;
                rmae('dev') = rmae_dev;
                rmae('test') = rmae_test;

                result('rmae') = rmae;

                mnrmae = containers.Map;
                mnrmae('train') = mnrmae_train;
                mnrmae('dev') = mnrmae_dev;
                mnrmae('test') = mnrmae_test;

                result('mnrmae') = mnrmae;

                mmnrmae = containers.Map;
                mmnrmae('train') = mmnrmae_train;
                mmnrmae('dev') = mmnrmae_dev;
                mmnrmae('test') = mmnrmae_test;

                result('mmnrmae') = mmnrmae;

                snrmae = containers.Map;
                snrmae('train') = snrmae_train;
                snrmae('dev') = snrmae_dev;
                snrmae('test') = snrmae_test;

                result('snrmae') = snrmae;

                qnrmae = containers.Map;
                qnrmae('train') = qnrmae_train;
                qnrmae('dev') = qnrmae_dev;
                qnrmae('test') = qnrmae_test;

                result('qnrmae') = qnrmae;

                out_results{n} = result;
                n=n+1;
                clear result;
                result = containers.Map;
            end    
        end
    end
    
    %Save BERT_runx_prediction_report.mat
    save([path, 'BERT_run' num2str(run) '_prediction_report.mat'], 'out_results');
    
end

function [srmse_phase, srmse_ampl] = fft_(signal1, signal2, plot_)
    time = 1:1:length(signal1);                         % Time Vector                               % Signal data in Time-Domain
    N = length(signal1);                                % Number Of Samples
    Ts = mean(diff(time));                              % Sampling Interval
    Fs = 1/Ts;                                          % Sampling Frequency
    Fn = Fs/2;                                          % Nyquist Frequency
    Fv = linspace(0, 1, fix(N/2)+1)*Fn;                 % Frequency Vector (For ‘plot’ Call)
    Iv = 1:length(Fv);                                  % Index Vector (Matches ‘Fv’)
    
    FT_Signal1 = fft(signal1)/N;                        % Normalized Fourier Transform Of Data
    FT_Signal2 = fft(signal2)/N;                        % Normalized Fourier Transform Of Data
    
    if plot_
        figure
        plot(Fv, abs(FT_Signal1(Iv))*2)
        hold on;
        plot(Fv, abs(FT_Signal2(Iv))*2)
    end
    
    %Mean squared errors
    %Phase
    tmp = (angle(FT_Signal1)'-angle(FT_Signal2)).^2;
    srmse_phase = sqrt(sum(tmp(:))/N)/std(FT_Signal1);
    
    %Phase
    tmp = (real(FT_Signal1)'-real(FT_Signal2)).^2;
    srmse_ampl = sqrt(sum(tmp(:))/N)/std(FT_Signal1);
    
end
        

