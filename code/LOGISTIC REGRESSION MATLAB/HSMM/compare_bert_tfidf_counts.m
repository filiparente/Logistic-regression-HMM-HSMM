%Compare tfidf counts to bert

dts = [1,3,4,6,12,24,48];
dws = [0,1,3,5,7]; %0,1,3,5,7
percentages = [0.8, 0.1, 0.1];
n_features = [768, 512, 256, 128, 64];

for i=1:length(dts)
    dt=num2str(dts(i));
    for j=1:length(dws)
        dw=num2str(dws(j));

        %LOAD BERT COUNTS
        data_bert=load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\datasets\server\' dt '.' dw '\dataset.mat']);

        all_obs_seq = data_bert.y;

        %Split train dev test
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
        window_size = str2num(dw);
        train_data_bert = all_obs_seq(1:lengths(1));
        dev_data_bert = all_obs_seq(lengths(1)+window_size+1:lengths(2));
        test_data_bert = all_obs_seq(lengths(2)+window_size+1:end);
        
        data_bert.y = [train_data_bert, dev_data_bert, test_data_bert];
        
        %disp(['Number of points in train dataset = ', num2str(length(train_data_bert))]);
        %disp(['Number of points in dev dataset = ', num2str(length(dev_data_bert))]);
        %disp(['Number of points in test dataset = ', num2str(length(test_data_bert))]);

        start_date = double(data_bert.start_date);
        end_date = double(data_bert.end_date);


        wholeSecs = floor(double(start_date)/1e9);
        fracSecs = double(start_date - uint64(wholeSecs)*1e9)/1e9;
        pd_start_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);
        wholeSecs = floor(double(end_date)/1e9);
        fracSecs = double(end_date - uint64(wholeSecs)*1e9)/1e9;
        pd_end_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);
        %disp(['Start date: ' datestr(pd_start_date) ' and end date: ' datestr(pd_end_date)])

        %embeddings = cell2mat(data.embeddings);
        obs_seq = double(data_bert.y'); %vertcat(embeddings.y); %counts
        %disp(['Observation sequence stats: min ' num2str(min(obs_seq)) ' max ' num2str(max(obs_seq)) ' mean ' num2str(mean(obs_seq)) ' std ' num2str(std(obs_seq))]);



        %LOAD TFIDF COUNTS
        dev_data_tfidf = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(1)) '\' dt '.' dw '\dev_dataset.mat']);
        test_data_tfidf = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(1)) '\' dt '.' dw '\test_dataset.mat']);
        train_data_tfidf = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(1)) '\' dt '.' dw '\train_dataset.mat']);

        data_tfidf.y = [train_data_tfidf.y, dev_data_tfidf.y, test_data_tfidf.y];

        for z=1:length(n_features)-1
            dev_data_tfidf2 = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(z+1)) '\' dt '.' dw '\dev_dataset.mat']);
            test_data_tfidf2 = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(z+1)) '\' dt '.' dw '\test_dataset.mat']);
            train_data_tfidf2 = load(['C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\server\n_features\' num2str(n_features(z+1)) '\' dt '.' dw '\train_dataset.mat']);

            data_tfidf2.y = [train_data_tfidf2.y, dev_data_tfidf2.y, test_data_tfidf2.y];
            data_tfidf2.X = [train_data_tfidf2.X; dev_data_tfidf2.X; test_data_tfidf2.X];
            assert(all(data_tfidf2.y==data_tfidf.y), 'something is wrong')
            assert(size(data_tfidf2.X,2)==n_features(z+1), 'something is wrong')
            
            clear dev_data_tfidf2
            clear test_data_tfidf2
            clear train_data_tfidf2
            clear data_tfidf2
        end
            
            
        %disp(['Number of points in train dataset = ', num2str(length(train_data_tfidf))]);
        %disp(['Number of points in dev dataset = ', num2str(length(dev_data_tfidf))]);
        %disp(['Number of points in test dataset = ', num2str(length(test_data_tfidf))]);

        start_date = double(train_data.start_date);
        end_date = double(test_data.end_date);


        wholeSecs = floor(double(start_date)/1e9);
        fracSecs = double(start_date - uint64(wholeSecs)*1e9)/1e9;
        pd_start_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);
        wholeSecs = floor(double(end_date)/1e9);
        fracSecs = double(end_date - uint64(wholeSecs)*1e9)/1e9;
        pd_end_date = datetime(wholeSecs,'ConvertFrom','posixTime','Format','yyyy.MM.dd HH:mm:ss.SSSSSSSSS') + seconds(fracSecs);
        %disp(['Start date: ' datestr(pd_start_date) ' and end date: ' datestr(pd_end_date)])

        %embeddings = cell2mat(data.embeddings);
        obs_seq = double(data_tfidf.y'); %vertcat(embeddings.y); %counts
        %disp(['Observation sequence stats: min ' num2str(min(obs_seq)) ' max ' num2str(max(obs_seq)) ' mean ' num2str(mean(obs_seq)) ' std ' num2str(std(obs_seq))]);
        
        if ~all(data_bert.y==data_tfidf.y)
            disp(['THERE IS A MAXIMUM DIFFERENCE OF ' num2str(max(abs(data_bert.y-data_tfidf.y))) '(which divided by the mean of y is ' num2str(max(abs(double(data_bert.y-data_tfidf.y)))/mean(double(data_bert.y)),4) ') IN DT ' dt ', DW ' dw ' in ' num2str(sum(data_bert.y~=data_tfidf.y)) ' element(s)!!!']);      
        else
            disp(['DT' dt ' DW ' dw ' is OK.']);
        end
            
    end
end
