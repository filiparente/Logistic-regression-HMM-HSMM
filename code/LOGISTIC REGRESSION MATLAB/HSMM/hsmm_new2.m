function [model, lkh, ll, ir]=hsmm_new2(params, MO,IterationNo,MT, tolerance, Vk, X, or_transf_A, or_lambdas, mode, weight_mode)
    %model has: lambdas, lambdas_tt, PAI,A,B,P,Qest, store_GAMMA, store_ALPHA

    %function [lambdas, lambdas_tt, PAI,A,B,PM,Qest,lkh, ll, ir]=hsmm_new(lambdas, PAI,A,B,PM, MO,IterationNo,MT, tolerance, Vk)
    % 
    % Author: Shun-Zheng Yu
    % Available: http://sist.sysu.edu.cn/~syu/Publications/hsmm_new.m
    % 
    % HSMM solve three fundamental problems for Hidden Semi-Markov Model using a new Forward-Backward algorithm
    % Usage: [PAI,A,B,P,Stateseq,Loglikelihood]=hsmm_new(PAI,A,B,P,O,IterationNo,MT)
    % MaxIterationNo=0: estimate StateSeq and calculate Loglikelihood only; 
    % MaxIterationNo>1: re-estimate parameters, estimate StateSeq and Loglikelihood.
    % First use [A,B,P,PAI,Vk,O,K]=hsmmInitialize(O,M,D,K) to initialize
    % 
    % Ref: Practical Implementation of an Efficient Forward-Backward Algorithm for an Explicit Duration Hidden Markov Model
    % by Shun-Zheng Yu and H. Kobayashi
    % IEEE Transactions on Signal Processing, Vol. 54, No. 5, MAY 2006, pp. 1947-1951 
    % 
    %  This program is free software; you can redistribute it and/or
    %  modify it under the terms of the GNU General Public License
    %  as published by the Free Software Foundation; either version
    %  2 of the License, or (at your option) any later version.
    %  http://www.gnu.org/licenses/gpl.txt
    %  
    %  Updated Nov.2014
    %
    %  N               Number of observation sequences
    %  MT              Lengths of the observation sequences: MT=(T_1,...,T_N)
    %  MO              Set of the observation sequences: MO=[O^1,...,O^N], O^n is the n'th obs seq.
    % mode - only applicable if the logistic regression is being used. 
    % if true, all features are used to estimate the weights of the
    % logistic regression; if false, only the features of the timestamps
    % associated with a transition, according to the state sequence
    % estimated by the EM algorithm.
    %++++++++ Markov Model +++++++++++
    
    lambdas = params.lambdas;
    PAI = params.PAI;
    A = params.A;
    B = params.B;
    P = params.P;
    if isfield(params, 'betas')
        model_betas = params.betas;
    end
    if isfield(params, 'pn')
        newton_param = params.pn;
    end
    
    M=length(PAI);               %The total number of states
    N=size(MO,2);                 %Number of observation sequences
    K=size(B,2);                 %The total number of observation values
    T=size(MO,1);
    size_A=size(A);
    W_iter = zeros(M, size(X,1)+1, M);
    if size(P,1)==1
        geometric=true; %flag that indicates that the duration distribution is a geometric distribution with one parameter per state
    else
        geometric=false; %otherwise, the duration distribution is a multinormal distribution given by a probability matrix for durationd d=1,...,D
    end

    if size_A(end)==M %stationary transition prob matrix -> HSMM
        statA = true;
    else
        statA = false; %dynamic transition prob matrix -> HSMM-LR
    end


    if ~geometric
        D=size(P,2);                 %The maximum duration of states BY D IN FUNCTION SINTAX
    else
        %Obtain the maximum allowed duration from the
        %duration parameters
        %duration such that the probability of P(D>dmax) is very small (e.g.: 0.01)
        %geometric dist: P(D>dmax)=1-P(D<=dmax)=1-(1-(1-p)^dmax) = 0.01
        %dmax=log(0.01)/log(1-p).
        PM = P;
        dmax = floor(log(0.001)/log(1-min(PM)));
        D = min(dmax, T);
    end
    original_D = D;
    
    ALPHA=zeros(M,D);
    aux_A = zeros(M,M,T);
    bmx=zeros(M,T);
    S=zeros(M,T);
    E=zeros(M,T);
    BETA=ones(M,D);
    Ex=ones(M,D);
    Sx=ones(M,D);
    GAMMA=zeros(M,1);
    Pest=zeros(M,D);

    if statA
        Aest=zeros(M,M);
    else
        Aest=zeros(M,M, T);
        n_features = size(X,1);
    end

    Best=zeros(M,K);
    PAIest=zeros(M,1);
    Qest=zeros(T, N);
    d=[1:D];
    store_GAMMA=zeros(M,T,N);

    if ~geometric
        store_ALPHA=zeros(M,D, T);
        store_ALPHAT=zeros(M,D, T);
    end

    lkh=zeros(1,N);
    ll = [];
    lambdas_tt = {}; %[];
    %final_W = [];

    ir1=max(1,IterationNo);

    for ir=1:ir1
        tic
        fprintf('Iteration nº: %d\n', ir);

        if geometric
            dmax = floor(log(0.001)/log(1-min(PM)));
            
            dmax = min(30, original_D+round(max((dmax-original_D)*0.5),0));
            D = dmax;
            d=[1:1:dmax];
        end
        d = [1:D];
        Pest=zeros(M,D);
        Best=zeros(M,K);
        PAIest=zeros(M,1);

        if statA
            Aest=zeros(M,M);
        else
            Aest=zeros(M,M, T);

            if IterationNo==0 %do not re estimate parameters, only estimate Qest
               new_A = zeros(M, M, T);
               
               for i=1:M
                   for j=setdiff(1:M,i)    
                       new_A(i,j,:) = exp(model_betas(i,:,j)*[X;ones(1,T)]-logsumexp(reshape(model_betas(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X;ones(1,T)])); 
                   end
               end
               A=new_A;
            end
        end

        for on=1:N	 % for each observation sequence

            O=MO(:,on);	 % the n'th observation sequence
            T=MT(on);        % the length of the n'th obs seq

            %    starttime=clock;
            %++++++++++++++++++     Forward     +++++++++++++++++
            %---------------    Initialization    ---------------
            %% REPLACED 

            if geometric
                %dmax = floor(log(0.001)/log(1-min(PM)));
                %D = dmax;
                %d = [1:D];

                clear P;
                for aux=1:M
                    P(aux,:) = (PM(aux)*(1-PM(aux)).^(d-1))/(1-(1-PM(aux)).^dmax); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
                    P(aux,:)=P(aux,:)/sum(P(aux,:));
                end
            end

            ALPHA(:)=0; ALPHA=repmat(PAI,1,D).*P;		%Equation (13)

            if ~geometric
                store_ALPHA(:,:,1) = ALPHA;
            end

            r=((poisspdf(O(1),lambdas)+1e-100)*sum(ALPHA,2));			%Equation (3)

            bmx(:,1)=(poisspdf(O(1),lambdas)+1e-100)./r;				%Equation (2)
            E(:)=0; E(:,1)=bmx(:,1).*ALPHA(:,1);		%Equation (5)

            S(:)=0; 

            if statA 
                S(:,1)=A'*E(:,1); 
            else
                S(:,1)=A(:,:,1)'*E(:,1);			%Equation (6)
            end

            lkh(on)=log(r);
            %---------------    Induction    ---------------
            for t=2:T
                ALPHA=[repmat(S(:,t-1),1,D-1).*P(:,1:D-1)+repmat(bmx(:,t-1),1,D-1).*ALPHA(:,2:D),S(:,t-1).*P(:,D)];		%Equation (12)

                if ~geometric
                    store_ALPHA(:,:,t) = ALPHA;
                end

                r=((poisspdf(O(t),lambdas)+1e-100)*sum(ALPHA,2));		%Equation (3)
                bmx(:,t)=(poisspdf(O(t),lambdas)+1e-100)./r;			%Equation (2)
                E(:,t)=bmx(:,t).*ALPHA(:,1);		%Equation (5)

                if statA 
                    S(:,t)=A'*E(:,t);
                else
                    S(:,t)=A(:,:,t)'*E(:,t);%A(:,:,t-1)'*E(:,t);				%Equation (6)
                end

                lkh(on)=lkh(on)+log(r);
            end

            if geometric
                store_ALPHA = ALPHA;
            end

            %++++++++ To check if the likelihood is increased ++++++++
            %if ir>1
            %    %ORIGINAL
            %    %    clock-starttime
            %    if (lkh-lkh1)/T<0.001
            %        break
            %    end
            %    
            %    %MINE
            %    %if(abs(lkh1(on))<abs(lkh(on))) %likelihood did not increase	
            %    %    disp('ERROR: likelihood did not increase in this iteration');
            %    %end
            %end

            %++++++++ Backward and Parameter Restimation ++++++++
            %---------------    Initialization    ---------------
            % MUDEI Aest=Aest+E(:,T)*ones(1,M);
            if statA
                Aest=Aest+E(:,T)*ones(1,M); %Since T_{T|T}(m,n) = E_{T}(m) a_{mn}
            else
                Aest(:,:,T-1)=Aest(:,:,T-1)+E(:,T)*ones(1,M);  %Since T_{T|T}(m,n) = E_{T}(m) a_{mn}
            end
            GAMMA=bmx(:,T).*sum(ALPHA,2);
            if IterationNo>0 && O(T)~=0
                Best(:,min(find(O(T)==Vk)))=Best(:,min(find(O(T)==Vk)))+GAMMA;
            end
            [~,Qest(T,on)]=max(GAMMA);
            store_GAMMA(:,T,on)=GAMMA;

            BETA=repmat(bmx(:,T),1,D);				%Equation (7)
            Ex=sum(P.*BETA,2);					%Equation (8)

            if statA
                Sx=A*Ex;
            else
                Sx=A(:,:,T-1)*Ex;						%Equation (9)
            end

            if ~geometric
                store_ALPHAT(:,:, T) = squeeze(store_ALPHA(:,:,end)).*BETA;
            end

            %---------------    Induction    ---------------
            for t=(T-1):-1:1
                %% for estimate of A
                % MUDEI Aest=Aest+E(:,t)*Ex';
                if statA
                    Aest=Aest+E(:,t)*Ex';
                else
                    if t>1 %ver se isto está bem. como estima em t=1?
                        Aest(:,:,t-1)=Aest(:,:,t-1)+E(:,t)*Ex';
                    end
                end

                %% for estimate of P
                Pest=Pest+repmat(S(:,t),1,D).*BETA;

                %% for estimate of state at time t
                GAMMA=GAMMA+E(:,t).*Sx-S(:,t).*Ex; %equation (16)
                GAMMA(GAMMA<0)=0;           % eleminate errors due to inaccurace of the computation.
                [~,Qest(t,on)]=max(GAMMA);
                store_GAMMA(:,t,on)=GAMMA;
                %% for estimate of B
                %Best(:,O(t))=Best(:,O(t))+GAMMA;
                if IterationNo>0 && O(t)~=0
                    Best(:,min(find(O(t)==Vk)))=Best(:,min(find(O(t)==Vk)))+GAMMA;
                end

                BETA=repmat(bmx(:,t),1,D).*[Sx,BETA(:,1:D-1)];	%Equation (14)
                [xx,yy]=find(BETA==Inf);
                if ~isempty(xx) && ~isempty(yy)
                    BETA(xx,yy) = 1e+308;
                end

                if ~geometric
                    store_ALPHAT(:,:,t) = squeeze(store_ALPHA(:,:,t)).*BETA;
                end

                Ex=sum(P.*BETA,2);			%Equation (8)
                % MUDEI Sx=A*Ex;
                if statA
                    Sx=A*Ex;
                else
                    if t>1  %   review
                        Sx=A(:,:,t-1)*Ex;	%Equation (9)
                    end
                end

                if(any(any(isnan(Aest))))
                    disp('erro');
                end
                if(any(any(isnan(Pest))))
                    disp('erro');
                end
                if(any(any(isnan(Best))))
                    disp('erro');
                end

            end

            Pest=Pest+repmat(PAI,1,D).*BETA;    %Since D_{1|T}(m,d) = \PAI(m) P_{m}(d) \Beta_{1}(m,d)
            PAIest=PAIest+GAMMA./sum(GAMMA);
        end % End for multiple observation sequences

        if ir>1
            if ((lkh-lkh1)/abs(lkh1))<0
                %Likelihood decreasing
                model = models(end);
                %if ~statA
                %    final_W = reshape(final_W, n_features+1,M^2,[]);
                %    model.betas_total = final_W;
                %end
                if geometric
                    P=model.PM;
                end
                ir = ir-1;
                disp('Likelihood decreased. Exiting...');
                break;
            end 

            if abs((lkh-lkh1)/abs(lkh1))<tolerance %the outer abs should not be possible, tho. before: ((lkh-lkh1)/abs(lkh1))<tolerance	
                %Stop if the increase in the likelihood is too small
                lkh1=lkh;
                ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration
                lambdas_tt{ir} = lambdas; %lambdas_tt = [lambdas_tt;lambdas];
                disp('Likelihood increase too small. Exiting...');
                break
            end     
         end

         lkh1=lkh;
         ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration

         if IterationNo>0            % re-estimate parameters
            models(ir).A = A;
            models(ir).lambdas = lambdas;
            models(ir).lambdas_tt = lambdas_tt;
            models(ir).s = PAI;
            models(ir).P = P;
            if geometric
                models(ir).PM=PM; %PM';
            end
            %models(ir).log_odds = log_odds_iter;
            models(ir).Qest = Qest;
            models(ir).B = B;
            models(ir).store_GAMMA = store_GAMMA;
            models(ir).store_ALPHA = store_ALPHA;

            %change it so that the intercept is first
            %if isempty(W_iter)
            %    W_iter=final_W(:,:,end);
            %end
            %W_iter = [W_iter(:,end,:),W_iter]; %[W_iter(end,:);W_iter];
            %W_iter = W_iter(:, 1:end-1,:); %W_iter(1:end-1,:);

            if ~statA
                models(ir).betas = W_iter;
                %models(ir).betas_total = final_W;
            end   
            Aest=Aest.*A; 

            idxs = find(round(sum(Best,2),1,'significant')==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
            idxs = [idxs; find(round(sum(Pest,2),1,'significant')==0)]; %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
            idxs = unique(idxs);

            if ~isempty(idxs)
                %first option: remove states -> we will not do this because we
                %use pruning
                %second option: set them to uniform

                if statA
                    Aest(idxs,:)=ones(length(idxs),M)/(M-1);	
                    Aest(idxs, idxs)=0;
                end
                Best(idxs,:)=ones(length(idxs),K)/K;
                Pest(idxs,:)=ones(length(idxs),D)/D;
            end

            clear idxs;

            if statA
                idxs = find(round(sum(Aest,2))==0);
                if ~isempty(idxs)

                    %SECOND OPTION: SET TO UNIFORM DISTRIBUTION
                    Aest(idxs,:,t)=ones(length(idxs),M)/(M-1);
                    for i=1:length(idxs)
                        Aest(idxs(i), idxs(i),t)=0;
                    end
                    clear idxs;
                end
            else

                for t=1:T
                    %idxs2 = find(round(sum(Aest(:,:,t),2))==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0. 
                    idxs = find(round(sum(Aest(:,:,t),2),1,'significant')==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.

                    if ~isempty(idxs)

                        %SECOND OPTION: SET TO UNIFORM DISTRIBUTION
                        Aest(idxs,:,t)=ones(length(idxs),M)/(M-1);
                        for i=1:length(idxs)
                            Aest(idxs(i), idxs(i),t)=0;
                        end
                        clear idxs;
                    end
                end
            end

            PAI=PAIest./sum(PAIest);
            A=Aest./repmat(sum(Aest,2),1,M);
            if(any(any(round(sum(A,2),1,'significant')~=1))) %check if transition matrix sums to 1, OK: 0, ERROR: 1
                disp('error: transition matrix EM does not sum to 1!');
            end
            %smoothing transition probabilities
            %A = smooth_transition_probs(A,M);

            B=Best./repmat(sum(Best,2),1,K);
            clear lambdas;
            for i=1:size(B,1)
                lambdas(i) = B(i,1:end)*Vk;
            end
            Pest=Pest.*P; 
            P=Pest./repmat(sum(Pest,2),1,D);

            if geometric
                den = P.*repmat([1:D]', 1,M)';
                den = sum(den,2);
                PM = 1./den;
                
                if size(PM,1)~=1
                    PM=PM';
                end
                    
                dmax = floor(log(0.001)/log(1-min(PM)));
                dmax = min(30, original_D+round(max((dmax-original_D)*0.5),0));
                d=[1:1:dmax];
                clear P;
                for aux=1:M
                    P(aux,:) = (PM(aux)*(1-PM(aux)).^(d-1))/(1-(1-PM(aux)).^dmax); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
                    P(aux,:)=P(aux,:)/sum(P(aux,:));
                end
            end

            if any(any(isnan(A)))
                disp('erro A');
            end
            if any(any(isnan(B)))
                disp('erro B');
            end
            if any(any(isnan(P)))
                disp('erro P');
            end
            if any(any(isnan(PAI)))
                disp('erro PAI');
            end

            lambdas_tt{ir} = lambdas; %lambdas_tt = [lambdas_tt;lambdas];

            if ~statA
                %After re-estimating the parameters (M-step)

                %UNSUPERVISED CASE: i do not have access to the state sequence,
                %instead, I have access to the state posteriors M
                %So we need M=n_states newton blocks 
                
                if ir==1
                    clear W_iter;
                    if strcmp(weight_mode, 'zeros')
                        W_iter = zeros(M, size(X,1)+1, M);
                    elseif strcmp(weight_mode, 'rand')
                        W_iter = randn(M, size(X,1)+1, M);
                    elseif strcmp(weight_mode, 'rand2')
                        W_iter = (1-(-1)).*rand(M, size(X,1)+1, M) + (-1);
                    end
                    clear W_iter2;
                    W_iter2 = zeros(M, size(X,1)+1, M);
                end
                
                %log_odds_iter = [];
                 clear Atemp;
                 Atemp = permute(A, [3 2 1]);
                 Anew = zeros(M, M-1, T);
                 for t=1:T
                     %remove the diagonals from the transition matrix since they have no information
                     Anew(:,:,t) = reshape(Atemp(t,~eye(M)), M-1, [])';
                 end
 
                 for nb=1:M
                     aux = Anew(nb:M:end,:,:); %A(nb:M:end,:,:);
                     if mode
                         %% SUGESTÃO DA ZITA: MSE DÁ MAIOR
                        [logit_model, ~] = logitMn(X, reshape(aux, M-1, length(X)), newton_param, squeeze(W_iter(nb, :, setdiff(1:M,nb)))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb));
                     else
                        %find indexes of the transitions from 1->2, 1->3 if nb=1, 2->1
                        %and 2->3 if nb=2 and 3->1 and 3->2 if nb=3
                        idxs_acum = [];
                        for jj=1:M
                            if jj~=nb
                                idxs_acum = [idxs_acum; strfind(Qest', [nb jj])'];
                            end
                        end
                        %idxs_acum = sort(idxs_acum);
                        %only transitions
                        %tic
                        [logit_model, ~] = logitMn(X(:,idxs_acum), reshape(aux(:,:,idxs_acum), M-1, length(idxs_acum)), newton_param, squeeze(W_iter(nb, :, setdiff(1:M,nb)))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb)));
                        %toc

                        %all       
                        %[logit_model, ~] = logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M-1, sum(Qest==nb))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb)));
                        %[logit_model, ~] = logitMn(X, reshape(At(nb:k:end,:,:), k, n));
                        %[~, log_odds] = logitMnPred(logit_model, X);
                        %[~, log_odds] = logitMnPred(logit_model, X(:,ss==nb));
                        %aux_A(nb, setdiff(1:M,nb),:)=log_odds;

                        %log_odds_iter = [log_odds_iter; log_odds];        
                     end
                     clear W;
                     W = logit_model.W;
                     %model.W contains the betas but the last row is the intercept, we will
                     %change to put it as the first row

                     %W = [W(end,:);W];
                     %W = W(1:end-1,:);
                     %W_iter = [W_iter, W];
                     %final_W=[final_W, W];

                     %logistic regression weights Bii are not used -> zeros
                     W_iter(nb, :, setdiff(1:M,nb)) = W;
                     %W_iter2(nb, :, setdiff(1:M,nb)) = logit_model.W2;
                 end

                %final_W = [final_W, W_iter];

                % Re-estimate the transition matrix with the estimated weights
%                 A = zeros(M, M, T);
% 
%                 for i=1:M
%                     for j=setdiff(1:M,i)
%                          A(i,j,:) = exp(W_iter(i,:,j)*[X;ones(1,T)]-logsumexp(reshape(W_iter(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X;ones(1,T)]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
%                         %before I've done this way, but its 20x more time consuming
%                         %and the results are the same (~10e-15 difference)
%                     end
%                 end
                
                A = zeros(M, M, T);
                
                %diferença <1e-14 e 98% + rapido (102s vs. 2s)
                for i=1:M
                    A(i,setdiff(1:M,i),:) = exp(squeeze(W_iter(i,:,setdiff(1:M,i)))'*[X;ones(1,T)]-repmat(logsumexp(reshape(W_iter(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X;ones(1,T)]), M-1, 1));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
                    A(i,i,:) = 0;
                end
                
                %try with W_iter2, check MSE between A and A2

                if(any(any(round(sum(A,2),1,'significant')~=1))) %check if transition matrix sums to 1, OK: 0, ERROR: 1
                    disp('error: transition matrix Logistic regression does not sum to 1!');
                end
            end

            if ~isempty(or_lambdas)
                %mse lambdas
                tmp = (or_lambdas - lambdas).^2;
                mse_lambdas(ir) = sum(tmp(:))/M;
            end

            if ~isempty(or_transf_A)
                %mse transition matrix
                tmp = (or_transf_A - A).^2;
                mse_A(ir) = sum(tmp(:))/numel(or_transf_A);
            end

            %if rem(ir, 10)==0
            %    figure
            %    plot(O)
            %    hold on
            %    plot(lambdas_tt{end}(Qest))
            %    ylabel('Observations')
            %    xlabel('t')
            %    lgd = legend({'True state sequence HSMM', 'Estimated state sequence HSMM'});
            %    lgd.Location = 'northeast';
            %    drawnow
            %end


%             models(ir).A = A;
%             models(ir).lambdas = lambdas;
%             models(ir).lambdas_tt = lambdas_tt;
%             models(ir).s = PAI;
%             models(ir).P = P;
%             if geometric
%                 models(ir).PM=PM';
%             end
%             %models(ir).log_odds = log_odds_iter;
%             models(ir).Qest = Qest;
%             models(ir).B = B;
%             models(ir).store_GAMMA = store_GAMMA;
%             models(ir).store_ALPHA = store_ALPHA;
% 
%             %change it so that the intercept is first
%             %if isempty(W_iter)
%             %    W_iter=final_W(:,:,end);
%             %end
%             %W_iter = [W_iter(:,end,:),W_iter]; %[W_iter(end,:);W_iter];
%             %W_iter = W_iter(:, 1:end-1,:); %W_iter(1:end-1,:);
% 
%             if ~statA
%                 models(ir).betas = W_iter;
%                 %models(ir).betas_total = final_W;
%             end   
         end

        toc;
    end

    if IterationNo>0
        model = models(end);
    else
        model.A = A;
        model.lambdas = lambdas;
        model.lambdas_tt = lambdas_tt;
        model.s = PAI;
        model.P = P;
        if geometric
            model.PM = PM';
            P = PM';
        end
        model.Qest = Qest;
        %model.betas = model_betas;
        model.B = B;
        model.store_GAMMA = store_GAMMA;
        model.store_ALPHA = store_ALPHA;
    end
    if ~statA
        model.betas = W_iter;
        %final_W = reshape(final_W, n_features+1,M^2,[]);
        %model.betas_total = final_W;
    end   
   

    if ~isempty(or_transf_A)
        if ~exist('mse_A')
            %mse transition matrix
            tmp = (or_transf_A - A).^2;
            mse_A(ir) = sum(tmp(:))/numel(or_transf_A);
        end

        figure
        subplot(3,1,1)
        plot(mse_A)
        ylabel('mse A')
        subplot(3,1,2)
        plot(mse_lambdas)
        ylabel('mse lambdas')
        subplot(3,1,3)
        plot(ll)
        ylabel('log-likelihood')
        xlabel('iterations')
    end
    disp(ir); %Print the number of iterations
end

        
