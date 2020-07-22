predictions_dir = 'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\results\real\1.0\hsmmLr\com_normalizacao\monte carlo\';
trainedmodel_path = 'C:\Users\Filipa\Desktop\LOGISTIC REGRESSION MATLAB\HSMM\results\real\1.0\hsmmLr\com_normalizacao\run1_results.mat';
dataset_path = 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\dt\1.0\new_cut_dataset.mat';

N=73;

%Load data
if ~exist('json_data','var')
    %fname = 'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\dt\1.0\new_cut_dataset.txt'; %new_small_dataset.txt'; %3.1\dataset.txt';%'C:\Users\Filipa\Desktop\Predtweet\results\3.1\dataset.txt'; 
    %json_data = loadjson(fname);
    data = load(dataset_path);
    data = data.r;
end

obs_seq = double(data.y'); %vertcat(embeddings.y); %counts  

percentages = [0.8, 0.1, 0.1];
window_size = 0;

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

train_obs_seq = obs_seq(1:lengths(1));
dev_obs_seq = obs_seq(lengths(1)+window_size+1:lengths(2));
test_obs_seq = obs_seq(lengths(2)+window_size+1:end);

dev_s = [];
test_s = [];
mse_dev = [];
mse_test = [];

%Load results
for i=1:N
    %data structure
    S = load([predictions_dir 'run' num2str(i) '_results.mat']);   
    
    %data structure
    dev_s = [dev_s; S.results.dev_est_state_seq];
    test_s = [test_s; S.results.test_est_state_seq];
    
    mse_dev = [mse_dev, S.results.mse_dev];
    mse_test = [mse_test, S.results.mse_test];
end

S = load([predictions_dir '_run_filtered_results.mat']);
mse_dev = [mse_dev, S.results.mse_dev];
mse_test = [mse_test, S.results.mse_test];
   
%PARAR AQUI PARA MOSTRAR MSE
%[val,idx]=min(mse_test);
%[val,idx]=min(mse_dev);

disp(['Mse dev: min ' num2str(min(mse_dev)) ' max ' num2str(max(mse_dev))  'mean ' num2str(mean(mse_dev)) ' std ' num2str(std(mse_dev))]); 
disp(['Mse test: min ' num2str(min(mse_test)) ' max ' num2str(max(mse_test))  'mean ' num2str(mean(mse_test)) ' std ' num2str(std(mse_test))]); 

%Load trained model to obtain lambdas
models = load(trainedmodel_path);

%Average predictions, mse dev and test
avg_dev = mean(models.results.hsmmLr_model.lambdas(dev_s));
avg_test = mean(models.results.hsmmLr_model.lambdas(test_s));

n_states=20;

counts=hist(dev_s,n_states);
[~,c] = max(counts);

avg_dev = models.results.hsmmLr_model.lambdas(c);
   
%Plot them against the true observations
FigH = figure('Position', get(0, 'Screensize'),'visible','off');
plot(dev_obs_seq, 'c') %test_obs_seq
hold on
plot(avg_dev, 'm') %avg_dev
hold on
lgd = legend({'True test observation sequence HSMM', 'Estimated test observation sequence HSMM'});
lgd.Location = 'northeast';
ylabel('State')
xlabel('t')
%drawnow

disp('MSE OBSERVATIONS (DEV)');
tmp = (avg_dev' - dev_obs_seq).^2;
results.mse_dev = sum(tmp(:))/numel(dev_obs_seq);
results.dev_est_state_seq = c;
results.dev_est_obs_seq = avg_dev;

F=getframe(FigH);
saveas(gcf,[predictions_dir '_dev_obs_avgpreds_seq.png']);

clear c;
counts=hist(test_s,n_states);
[~,c] = max(counts);

avg_test = models.results.hsmmLr_model.lambdas(c);

FigH = figure('Position', get(0, 'Screensize'),'visible','off');
plot(test_obs_seq, 'c') %test_obs_seq
hold on
plot(avg_test, 'm')
hold on
lgd = legend({'True test observation sequence HSMM', 'Estimated test observation sequence HSMM'});
lgd.Location = 'northeast';
ylabel('State')
xlabel('t')
%drawnow

disp('MSE OBSERVATIONS (TEST)');
tmp = (avg_test' - test_obs_seq).^2;
results.mse_test = sum(tmp(:))/numel(test_obs_seq);
results.test_est_state_seq = c;
results.test_est_obs_seq = avg_test;

F=getframe(FigH);
saveas(gcf,[predictions_dir '_test_obs_avgpreds_seq.png']);

FigH = figure('Position', get(0, 'Screensize'),'visible','off');
plot(obs_seq, 'c') %test_obs_seq
hold on
plot([models.results.hsmmLr_model.lambdas(models.results.hsmmLr_model.Qest),avg_dev, avg_test], 'm')
hold on
lgd = legend({'True test observation sequence HSMM', 'Estimated test observation sequence HSMM'});
lgd.Location = 'northeast';
ylabel('State')
xlabel('t')
%drawnow

F=getframe(FigH);
saveas(gcf,[predictions_dir '_all_obs_avgpreds_seq.png']);

            
save([predictions_dir '_run_filtered_results.mat'],'results');
            
            
