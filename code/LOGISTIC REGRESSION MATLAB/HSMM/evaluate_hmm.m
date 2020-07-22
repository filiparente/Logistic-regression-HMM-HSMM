function [mse, next_state_posteriors, predicted_count, out_A] = evaluate_hmm(model, obs_seq, map, prev_state_posteriors, fluctuations)
    %% DEV RESULTS
    %dev
    %A = A_train(:,:,end);
    %test
    %A = new_A_dev(:,:,end);
    %Predict future observations
    n_states = length(model.lambdas);
    
    if ~prev_state_posteriors
        prev_state_posteriors = squeeze(model.store_GAMMA(:,end));
    end
    
%     size_A = size(model.A); 
%     if size_A(end)>n_states %HSMM_LR
%         %Get transition matrix from features
%         new_A = zeros(n_states, n_states, size(obs_seq,1));
%         for g=1:n_states
%             for k=1:n_states
%                 %column = get_columns_(M,i,j);
%                 for t=1:size(obs_seq,1)
%                     new_A(g,k,t) = exp(model.betas(g,:,k)*[X(:,t);1]-logsumexp(reshape(model.betas(g,:,setdiff(1:n_states,g)), size(X,1)+1, n_states-1)'*[X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
%                     %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
%                 end
%             end 
%         end
%         
%         [predicted_count, next_state_posteriors] = predict_future(prev_state_posteriors, cat(3,A, new_A), model.lambdas, obs_seq, n_states, map);
%  
%         out_A = new_A(:,:,end);
%     else
        [predicted_count, next_state_posteriors] = predict_future(prev_state_posteriors, model.A, model.lambdas, obs_seq, n_states, map, fluctuations);   
 
        out_A = model.A;
%     end

    tmp = (predicted_count'-obs_seq).^2;
    mse = sum(tmp(:))/numel(obs_seq);
    
    
end