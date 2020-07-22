function [mse, prev_alpha, predicted_count, out_A] = evaluate_hsmm(model, A, X, obs_seq, map, last_obs, prev_alpha, plot_, fluctuations)
    %dev
    %A = A_train(:,:,end);
    %test
    %A = new_A_dev(:,:,end);
    %% DEV RESULTS
    %Predict future observations
    n_states = length(model.lambdas);
    original_P = [];
    
    size_A = size(model.A); 
    
    if ~prev_alpha
        prev_alpha = squeeze(model.store_ALPHA(:,:,end));
    end
        
    if size_A(end)>n_states %HSMM_LR
        %Get transition matrix from features
        new_A = zeros(n_states, n_states, size(obs_seq,1));
%         for g=1:n_states
%             for k=setdiff(1:n_states,g) 
%                 %column = get_columns_(M,i,j);
%                 for t=1:size(obs_seq,1)
%                     new_A(g,k,t) = exp(model.betas(g,:,k)*[X(:,t);1]-logsumexp(reshape(model.betas(g,:,setdiff(1:n_states,g)), size(X,1)+1, n_states-1)'*[X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
%                     %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
%                 end
%             end 
%         end
      
        %diferença <1e-14 e 98% + rapido (102s vs. 2s)
        for i=1:n_states
            new_A(i,setdiff(1:n_states,i),:) = exp(squeeze(model.betas(i,:,setdiff(1:n_states,i)))'*[X;ones(1,size(obs_seq,1))]-repmat(logsumexp(reshape(model.betas(i,:,setdiff(1:n_states,i)), size(X,1)+1, n_states-1)'*[X;ones(1,size(obs_seq,1))]), n_states-1, 1));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
            new_A(i,i,:) = 0;
        end
                                                   %dev_or_transf_A
        [predicted_count, next_alpha] = predict_future2(prev_alpha, cat(3,A, new_A), model.lambdas, obs_seq, n_states, map, model.P, original_P,last_obs, plot_, fluctuations);        
        out_A = new_A(:,:,end);
    
    else 
        %[Vk, obs_seq, state_seq] = hsmmSample(ss,model.A,model.P,model.B, model.lambdas, size(dev_obs_seq,1),size(dev_obs_seq,2));
        %[predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, model.A, model.lambdas, dev_obs_seq, n_states, map);
        [predicted_count, next_alpha] = predict_future2(prev_alpha, model.A, model.lambdas, obs_seq, n_states, map, model.P, original_P, last_obs, plot_, fluctuations);
        out_A = model.A;
    end
    
    prev_alpha = next_alpha;

    tmp = (predicted_count'-obs_seq).^2;
    mse = sum(tmp(:))/numel(obs_seq);
    rmse = sqrt(mse);
    nrmse = rmse/std(obs_seq);
    
end