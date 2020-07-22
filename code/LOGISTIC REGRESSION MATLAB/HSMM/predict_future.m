%% PREDICTIONS FOR HMMS
function [predicted_count, state_posteriors] = predict_future(state_posteriors, A, lambdas, obs_seq, n_states, map, fluctuations)
    prev_state_posteriors = state_posteriors;
    T=size(obs_seq,1);
    sizeA = size(A);
    
    if sizeA(end)==n_states %stationary transition_matrix
        A = repmat(A, 1,1, T);
    end
    
    for t=1:T
        next_state_posteriors = A(:,:,t)'*prev_state_posteriors;
            
        %Normalize
        next_state_posteriors = next_state_posteriors./sum(next_state_posteriors);
       
        %Predict count
        if map
          
            %argmax das state posteriors aplicado ao lambda correspondente,
            %estimado no treino
            [~, idx] = max(next_state_posteriors);
            if fluctuations
                predicted_count(t) = poissrnd(lambdas(idx))';
            else
                predicted_count(t) = lambdas(idx)';
            end
        else
            
            %Média ponderada das state posteriors com os lambdas estimados
            %no treino
            if fluctuations
                predicted_count(t) = round(next_state_posteriors'*poissrnd(lambdas)');
            
            else
               predicted_count(t)= round(next_state_posteriors'*lambdas');
            end
            
        end
        
        %Adjust
        for i=1:n_states
            %+1e-100 avoids the cases where the observation is not well
            %explained by any of the trained lambdas -> therefore the
            %prev_state_posteriors give all zeros, and after normalizing
            %we get Nans
            prev_state_posteriors(i) = next_state_posteriors(i)*(poisspdf(obs_seq(t),lambdas(i))+1e-100);
        end
        prev_state_posteriors = prev_state_posteriors./sum(prev_state_posteriors);
        
    end
end



%% OLD CODE
% function [predicted_count, state_posteriors] = predict_future(state_posteriors, last_state_dur, A, lambdas, obs_seq, n_states, map, D)
%     size_A = size(A);
%     length_dur = size(D,2);
%     prev_state_posteriors = state_posteriors;
%     next_state_posteriors = state_posteriors;
%     %rr = zeros(800,2);
%     %for t=1:800
%         %auxx=squeeze(store_ALPHAT(:,t,:));
%         %[m,n] = max(auxx(:));
%         %[rr(t,1), rr(t,2)] = ind2sub(size(auxx), n);
%     %end
%     [~,n] = max(last_state_dur(:));
%     [state, dur] = ind2sub(size(last_state_dur), n);
%     
%     dur = dur-1;
%     idx = state;
%     
%     for t=1:size(obs_seq,1)
%         if dur==0
%             if size_A(end)>n_states %HSMM_LR
%                 next_state_posteriors = A(:,:,t)'*prev_state_posteriors;
%                 
%             else
%                 next_state_posteriors = A'*prev_state_posteriors;
%             end
%             %Normalize
%             next_state_posteriors = next_state_posteriors./sum(next_state_posteriors);
%             %next_state_posteriors_d = repmat(next_state_posteriors, 1,size(D,2)).*D;
%         end
% 
%         %Predict count
%         if map
%             if dur~=0
%                 predicted_count(t) = poissrnd(lambdas(idx))';
%                 dur = dur-1;
%             else
%                 %argmax das state posteriors aplicado ao lambda correspondente,
%                 %estimado no treino
%                 
%                 [~, idx] = max(abs(next_state_posteriors-state_posteriors));
%                 %[~, idx] = max(next_state_posteriors);
% 
%                 %Deal with durations
%                 %L = 1:1:size(D,2);
%                 %dur = D(idx,:)*L'; 
%                 %[~,dur] = min(abs(dur-L));
%                 %[~,dur] = max(D(idx,:));
%                 
%                 aux = (D.*repmat([1:1:length_dur], n_states,1))./repmat(sum(D,1), n_states,1);
%                 
%                 dur_low = sum((sum(D.*repmat([1:1:length_dur], n_states,1),2)-std(aux,0,2)).*next_state_posteriors);
%                 dur_high = sum((sum(D.*repmat([1:1:length_dur], n_states,1),2)+std(aux,0,2)).*next_state_posteriors);
%                 avg_dur = sum(sum(D.*repmat([1:1:length_dur], n_states,1),2).*next_state_posteriors);
%                 
%                 dur_low = floor(dur_low);
%                 dur_high = floor(dur_high);
%                 avg_dur = floor(avg_dur);
%                 
%                 dur = avg_dur;%dur_high;
%                 
%                 %[m,n] = max(next_state_posteriors_d(:));
%                 %[idx, dur] = ind2sub(size(next_state_posteriors_d), n);
%                 
%                 predicted_count(t) = poissrnd(lambdas(idx))';
%                 dur = dur-1;
%                 prev_state_posteriors = next_state_posteriors;
%      
%             end
%         else
% 
%             %Média ponderada das state posteriors com os lambdas estimados
%             %no treino
% 
%             %Deal with durations
%             L = 1:1:size(D,2);
%             dur = next_state_posteriors'*D;
%             dur = dur*L';
% 
%             predicted_count(t) = next_state_posteriors'*poissrnd(lambdas)';
%             dur=dur-1;
%         end
%         %Adjust
%         for i=1:n_states
%             state_posteriors(i) = next_state_posteriors(i)*poisspdf(obs_seq(t),lambdas(i));
%         end
%         state_posteriors = state_posteriors./sum(state_posteriors);
%         
%         %[prob, idx2] = max(state_posteriors);
%         %if(idx2~=idx && prob>0.7)
%         %    dur=0;
%         %end
% 
%     end
% end