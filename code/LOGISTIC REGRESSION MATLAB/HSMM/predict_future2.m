function [pred_obs, next_alpha] = predict_future2(prev_alpha, A, lambdas, obs_seq, n_states, map, D, original_P, last_obs, plot_, fluctuations)
    sizeA = size(A);
    T=size(obs_seq,1);
    M=n_states;
    bmx=zeros(M,T);
    S=zeros(M,T);
    S2=zeros(M,T);
    E=zeros(M,T);
    if sizeA(end)==n_states %stationary transition_matrix
        A = repmat(A, 1,1, T+1);
    end
    
    list_d = [];
    list_s = [];
    
   %% SZY CODE
   P=D;
   D=size(P,2);
    
   ALPHA=prev_alpha;
    
   %estimate last state q_T using the filtered state posterior
   %GAMMA_T|T(m)= bm*(oT)GAMMA_T|T-1(m)
   %where GAMMA_T|T-1(m)=sum_d ALPHA_T|T-1(m,d)
   
   %GAMMA=bmx(:,T).*sum(ALPHA,2);
   
   %However, here, as we do not have the current observation available,
   %only the previous ones, we will use the predicted state posterior
   %GAMMA_T|T-1(m) instead of the filtered state posterior GAMMA_T|T(m)
 
    r=((poisspdf(last_obs,lambdas)+1e-100)*sum(ALPHA,2));			%Equation (3)

    bmx(:,1)=(poisspdf(last_obs,lambdas)+1e-100)./r;				%Equation (2)
    E(:)=0; E(:,1)=bmx(:,1).*ALPHA(:,1);		%Equation (5)
    % MUDEI
    S(:)=0; S(:,1)=A(:,:,1)'*E(:,1);			%Equation (6)

    if plot_
        h = figure;
        axis tight manual % this ensures that getframe() returns a consistent size
        filename = 'testAnimated.gif';
        FigH = figure('Position', get(0, 'Screensize'), 'visible','on');
        plot(obs_seq, 'c') %test_obs_seq
        hold on
    end
    
    %---------------    Induction    ---------------
    for t=2:T+1             
       
        %if t==2
            %in the beggining, make transitions count more
        %    ALPHA=[0.9.*repmat(S(:,t-1),1,D-1).*P(:,1:D-1)+0.1.*repmat(bmx(:,t-1),1,D-1).*ALPHA(:,2:D),S(:,t-1).*P(:,D)];
        %else
            ALPHA=[repmat(S(:,t-1),1,D-1).*P(:,1:D-1)+repmat(bmx(:,t-1),1,D-1).*ALPHA(:,2:D),S(:,t-1).*P(:,D)];
        %end
        
        %Equation (12)
        GAMMA = sum(ALPHA,2);
        
        %if t==2
        %    aux=sum(abs(prev_alpha-ALPHA)./(prev_alpha+1e-10),2);
        %    GAMMA=aux./sum(aux);
        %end
        
        %Sanity check
        if round(sum(GAMMA), 3, 'significant')~=1.00
           sum_ = sum(GAMMA);
           GAMMA = GAMMA./sum_;
           ALPHA = ALPHA./repmat(sum(ALPHA,2),1,D);
           ALPHA = ALPHA.*repmat(GAMMA, 1, D);
           %next_alpha = next_alpha./repmat(next_gamma'.*sum(next_alpha,2), 1, length_dur);
        end
    
            
       %Predict next observation
        if map
            [~, state(t-1)] = max(GAMMA);
            
            if fluctuations
                pred_obs(t-1) = poissrnd(lambdas(state(t-1))); 
            else
                pred_obs(t-1) = lambdas(state(t-1));
            end
        else
            %Weighted average
            if fluctuations
                pred_obs(t-1) = round(GAMMA'*poissrnd(lambdas)'); 
            else
                pred_obs(t-1) = round(GAMMA'*lambdas');
            end
            
        end
        
        if plot_
            scatter(t-1, pred_obs(t-1), 'm', '.')
            drawnow
            if t>2
                plot([t-2 t-1],[pred_obs(t-2) pred_obs(t-1)], 'b')
            end
        
        
            % Capture the plot as an image 
            frame = getframe(FigH); 
            im = frame2im(frame); 
            [imind,cm] = rgb2ind(im,256); 

            % Write to the GIF File 
            if t==2 
                imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
            else 
                imwrite(imind,cm,filename,'gif','WriteMode','append'); 
            end 
        end
        
        %if t==2
        %    [~, state(t-1)] = max(GAMMA);
        %    
        %    pred_obs(t-1) = poissrnd(lambdas(state(t-1)));
        %    
        %    %r=((poisspdf(pred_obs(t-1),lambdas)+1e-100)*sum(ALPHA,2));			%Equation (3)
        %    %bmx_aux=(poisspdf(pred_obs(t-1),lambdas)+1e-100)./r;				%Equation (2)
        %    %ALPHA=ALPHA.*repmat(bmx_aux', 1, D);
        %    [~,d] = max(ALPHA(state(t-1),:));
        %    
        %    %change durations
        %    val = ALPHA(state(t-1),d);
        %    val2 = ALPHA(state(t-1),d-1);
        %    ALPHA(state(t-1),d-1)=val;
        %    ALPHA(state(t-1),d)=val2;  
        %end
        
        [~,aux] = max(ALPHA(:));
        [prev_s,prev_d]=ind2sub(size(ALPHA), aux);
        list_d = [list_d;prev_d];
        list_s = [list_s; prev_s];
        
        %if length(list_d)>1 && (list_d(end)==2 && all(diff(list_d)==-ones(length(list_d)-1,1))) && all(diff(list_s)==zeros(length(list_s)-1,1))   
        %    val = ALPHA(list_s(1),list_d(end));
        %    val2 = ALPHA(list_s(1),list_d(end)-1);
        %    ALPHA(list_s(1),list_d(end)-1)=val;
        %    ALPHA(list_s(1),list_d(end))=val2;  
            
            
            %GAMMA(list_s(1)) = 0;
            %GAMMA = GAMMA./sum(GAMMA);
            
            %aux_ALPHA=ALPHA;
            %aux_ALPHA(list_s(1),:) = 0;
            %[~,aux] = max(aux_ALPHA(:));
            %[prev_s,prev_d]=ind2sub(size(ALPHA), aux);
            
        %    list_d = [];
        %    list_s = [];  
            
        %    list_d = [list_d;prev_d];
        %    list_s = [list_s; prev_s];
        %end
        
        r=((poisspdf(obs_seq(t-1),lambdas)+1e-100)*sum(ALPHA,2));		%Equation (3)
        bmx(:,t)=(poisspdf(obs_seq(t-1),lambdas)+1e-100)./r;			%Equation (2)
        E(:,t)=bmx(:,t).*ALPHA(:,1);		%Equation (5)
        % MUDEI S(:,t)=A'*E(:,t);
        S(:,t)=A(:,:,t)'*E(:,t);%A(:,:,t-1)'*E(:,t);				%Equation (6)
        S2(:,t)=mean(A,3)'*E(:,t);
        prev_ALPHA = ALPHA;
    end

    next_alpha=ALPHA;

    if plot_
        hold on
        lgd = legend({'True dev observation sequence HSMM', 'Estimated dev observation sequence HSMM'});
        lgd.Location = 'northeast';
        ylabel('State')
        xlabel('t')
    end

    %% MY CODE
%     length_dur = size(D,2);    
% 
%     for t=1:size(obs_seq,1)
%         %Compute emissions probability
%         b = poisspdf(last_obs, lambdas)';
% 
%         den = sum(sum(prev_alpha.*repmat(b, 1, length_dur)));
% 
%         for i=1:n_states
%             bx(i) = b(i)/den;
%         end
% 
%         %Re-estimate state posteriors adjusted by previous observation
%         for m=1:n_states
%             if t>1
%                 sum_ = 0;
%                 for n=1:n_states
%                     sum_ = sum_ + prev_alpha(n,1)*bx(n)*A(n,m,t);
%                 end
%                 [~,aux] = max(prev_alpha(:));
%                 [~,prev_d]=ind2sub(size(prev_alpha), aux);
%                 
%                 for d=1:length_dur
%                     
%                     if d+1>length_dur %|| (bx(m)>1 && prev_d==1)
%                         next_alpha(m,d) = sum_*D(m,d);
%                     else
%                         next_alpha(m,d) = sum_*D(m,d) + bx(m)*prev_alpha(m,d+1);
%                     end
%                 end
%                 next_gamma(m) = sum(next_alpha(m,:));
%             else
%                 next_gamma(m) = sum(prev_alpha(m,:));
%                 next_alpha = prev_alpha;
%             end
%         end
%         
%         %Sanity check
%         if round(sum(next_gamma), 3, 'significant')~=1.00
%            sum_ = sum(next_gamma);
%            next_gamma = next_gamma./sum_;
%            next_alpha = next_alpha./repmat(sum(next_alpha,2),1,length_dur);
%            next_alpha = next_alpha.*repmat(next_gamma', 1, length_dur);
%            %next_alpha = next_alpha./repmat(next_gamma'.*sum(next_alpha,2), 1, length_dur);
%         end
% 
%         %Predict next observation
%         if map
%             [~, state(t)] = max(next_gamma);
%             pred_obs(t) = poissrnd(lambdas(state(t)));
%         else
%             %Weighted average
%             pred_obs(t) = next_gamma*poissrnd(lambdas)';
%         end
% 
%         %Update variables
%         prev_alpha =  next_alpha;
%         if t>1
%             last_obs = obs_seq(t);
%         end
%     end 
    
end