%% REPLACED                                                        store_ALPHAT
function [model, lambdas, lambdas_tt, PAI,A,B,P,Qest, store_GAMMA, store_ALPHA, lkh, ll, ir]=hsmm_new_(lambdas, PAI,A,B,P,MO,IterationNo,MT, tolerance, Vk, X, or_transf_A, or_lambdas, model_betas)
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
%++++++++ Markov Model +++++++++++
M=length(PAI);               %The total number of states
N=size(MO,2);                 %Number of observation sequences
K=size(B,2);                 %The total number of observation values
T=size(MO,1);

if size(P,1)==1
    geometric=true; %flag that indicates that the duration distribution is a geometric distribution with one parameter per state
else
    geometric=false; %otherwise, the duration distribution is a multinormal distribution given by a probability matrix for durationd d=1,...,D
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

%----------------------------------------------------


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
% MUDEI
Aest=zeros(M,M, T);
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
lambdas_tt = {};
final_W = [];

n_features = size(X,1);

ir1=max(1,IterationNo);

for ir=1:ir1
    tic
    fprintf('Iteration nº: %d\n', ir);
    if geometric
        dmax = floor(log(0.001)/log(1-min(PM)));
        D = dmax;
    end
    d = [1:D];
    Pest=zeros(M,D);
    Aest=zeros(M,M, T);
    Best=zeros(M,K);
    PAIest=zeros(M,1);
    %Aest(:)=0;
    %Pest(:)=0;
    %Best(:)=0;
    %PAIest(:)=0;
    if IterationNo==0 %do not re estimate parameters, only estimate Qest
       new_A = zeros(M, M, T);
       
       for i=1:M
           %columns = get_columns_(M, i);
           %columns = columns(columns~=get_columns_(M,i,i));
           for j=setdiff(1:M,i)
               %column = get_columns_(M,i,j);     
               for t=1:T                 
                   new_A(i,j,t) = exp(model_betas(i,:,j)*[X(:,t);1]-logsumexp(reshape(model_betas(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
                   %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
               end
           end
       end
       A=new_A;
    end
    
   
    for on=1:N	 % for each observation sequence

        
        O=MO(:,on);	 % the n'th observation sequence
        T=MT(on);        % the length of the n'th obs seq
        
        %    starttime=clock;
        %++++++++++++++++++     Forward     +++++++++++++++++
        %---------------    Initialization    ---------------
        %% REPLACED 
        
        if geometric
            dmax = floor(log(0.001)/log(1-min(PM)));
            D = dmax;
            d = [1:D];
            
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
        % MUDEI
        S(:)=0; S(:,1)=A(:,:,1)'*E(:,1);			%Equation (6)
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
            % MUDEI S(:,t)=A'*E(:,t);
            S(:,t)=A(:,:,t)'*E(:,t);%A(:,:,t-1)'*E(:,t);				%Equation (6)
            lkh(on)=lkh(on)+log(r);
        end
        
        if geometric
            store_ALPHA = zeros(M, D, 1);
            store_ALPHA = ALPHA;
            %store_ALPHA(:,:,end)
        end
            
        
        %++++++++ Backward and Parameter Restimation ++++++++
        %---------------    Initialization    ---------------
        % MUDEI Aest=Aest+E(:,T)*ones(1,M);
        Aest(:,:,T-1)=Aest(:,:,T-1)+E(:,T)*ones(1,M);  %Since T_{T|T}(m,n) = E_{T}(m) a_{mn}
        GAMMA=bmx(:,T).*sum(ALPHA,2);
        
        if IterationNo>0 && O(T)~=0
            Best(:,min(find(O(T)==Vk)))=Best(:,min(find(O(T)==Vk)))+GAMMA;
        end
        [~,Qest(T,on)]=max(GAMMA);
        store_GAMMA(:,T,on)=GAMMA;
        
        BETA=repmat(bmx(:,T),1,D);				%Equation (7)
        
        if ~geometric
            store_ALPHAT(:,:, T) = squeeze(store_ALPHA(:,:,end)).*BETA;
        end

        aux5=BETA;
        Ex=sum(P.*BETA,2);					%Equation (8)
        % MUDEI Sx=A*Ex
        Sx=A(:,:,T-1)*Ex;						%Equation (9)
        kk = 1;
        %---------------    Induction    ---------------
        for t=(T-1):-1:1
            %% for estimate of A
            % MUDEI Aest=Aest+E(:,t)*Ex';
            if t>1
                Aest(:,:,t-1)=Aest(:,:,t-1)+E(:,t)*Ex';
            end
            %% for estimate of P
            Pest=Pest+repmat(S(:,t),1,D).*BETA;
            if t>T-1-D
                aux1=(repmat(S(:,t),1,D).*BETA);
                aux2 = aux1.*P;
                [m n] = max(aux2(:));
                [last_state(kk), last_dur(kk)] = ind2sub(size(aux2), n);
                kk = kk+1;
            end
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
            
            Ex=sum(P.*BETA,2);					%Equation (8)
            % MUDEI Sx=A*Ex;
            if t>1
                Sx=A(:,:,t-1)*Ex;						%Equation (9)
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
            P=model.PM;
            ir = ir-1;
            disp('Likelihood decreased. Exiting...');
            break;
        end
        
        if abs((lkh-lkh1)/abs(lkh1))<tolerance %the outer abs should not be possible, tho. maybe the likelihood is different? change it or use parameter tolerance.
            %Stop if the increase in the likelihood is too small
            lkh1=lkh;
            ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration
            lambdas_tt{ir} = lambdas;
            disp('Likelihood increase too small. Exiting...');
            break
        end
    end
    lkh1=lkh;
    ll = [ll;lkh]; %add the likelihood for the N observations sequences in the current iteration
    
    if IterationNo>0            % re-estimate parameters
        Aest=Aest.*A;  
        idxs = find(round(sum(Best,2),1,'significant')==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
        idxs = [idxs; find(round(sum(Pest,2),1,'significant')==0)]; %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
        idxs = unique(idxs);
        %Best(idxs,:)=ones(length(idxs),K)/K;
        %Pest(idxs,:)=ones(length(idxs),D)/D;
        
        %Aest(idxs,:)=[];%ones(M,1)/M;
        %Aest(:,idxs)=[];
        if ~isempty(idxs)
            %Aest(idxs,:,:)=[];
            %Aest(:,idxs,:)=[];
            %Best(idxs,:)=[];
            %Pest(idxs,:)=[];
            %P(idxs,:)=[];
            %PAIest(idxs)=[];
            %for i=1:length(Qest)
            %    Qest(i) = Qest(i)-sum(idxs<Qest(i));
            %end
      
            %M=M-length(idxs);
            %ALPHA=zeros(M,D);
            %bmx=zeros(M,T);
            %S=zeros(M,T);
            %E=zeros(M,T);
            %BETA=ones(M,D);
            %Ex=ones(M,D);
            %Sx=ones(M,D);
            %GAMMA=zeros(M,1);
            Best(idxs,:)=ones(length(idxs),K)/K;
            Pest(idxs,:)=ones(length(idxs),D)/D;
        end
        clear idxs;
        
        for t=1:T
            %idxs2 = find(round(sum(Aest(:,:,t),2))==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0. 
            idxs = find(round(sum(Aest(:,:,t),2),1,'significant')==0); %idxs of the states that do not appear in the estimated state sequence, so their probability distribution in the transition matrix sums to 0.
            
            if ~isempty(idxs)
                %FIRST OPTION: REMOVE
                %remove them from all estimated parameters

                %Aest(idxs,:)=[];%ones(M,1)/M;
                %Aest(:,idxs)=[];
                %Best(idxs,:)=[];
                %Pest(idxs,:)=[];
                %P(idxs,:)=[];
                %PAIest(idxs)=[];
                %M=M-length(idxs);
                %ALPHA=zeros(M,D);
                %bmx=zeros(M,T);
                %S=zeros(M,T);
                %E=zeros(M,T);
                %BETA=ones(M,D);
                %Ex=ones(M,D);
                %Sx=ones(M,D);
                %GAMMA=zeros(M,1);

                %SECOND OPTION: SET TO UNIFORM DISTRIBUTION
                Aest(idxs,:,t)=ones(length(idxs),M)/(M-1);
                for i=1:length(idxs)
                    Aest(idxs(i), idxs(i),t)=0;
                end
                %Best(idxs,:)=ones(length(idxs),K)/K;
                %Pest(idxs,:)=ones(length(idxs),D)/D;

                clear idxs;
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
        %% REPLACED 
        if geometric
            Pest1 = Pest;
            Pest=Pest1.*P; 
            Pest=Pest./repmat(sum(Pest,2),1,D); 
            den = Pest.*repmat([1:D]', 1,M)';
            den = sum(den,2);
            PM1 = 1./den;
            %% OU
            %Pest=Pest1.*P; den = Pest.*repmat([1:D]', 1,M)'; den = sum(den,2);
            %PM2 = sum(store_GAMMA(:,1:end-1), 2)./den;
            %PM3 = den;
            %for kk=1:M
            %    PM4(kk) = (Pest(kk,:)/sum(Pest(kk,:)))*[1:D]';
            %end
            %PM4(PM4>1)=1-1e-16;
            
            PM=PM1;
            dmax = floor(log(0.001)/log(1-min(PM)));
            d=[1:1:dmax];
            clear P;
            for aux=1:M
                P(aux,:) = (PM(aux)*(1-PM(aux)).^(d-1))/(1-(1-PM(aux)).^dmax); %poisspdf(d,PM(aux)+1e-100); %geopdf(d,PM(aux)); 
                P(aux,:)=P(aux,:)/sum(P(aux,:));
            end
        else
            Pest=Pest.*P;   P=Pest./repmat(sum(Pest,2),1,D);
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
        
        lambdas_tt{ir} = lambdas;
        
        %After re-estimating the parameters (M-step)
        
        %UNSUPERVISED CASE: i do not have access to the state sequence,
        %instead, I have access to the state posteriors M
        %So we need M=n_states newton blocks 
        
        clear W_iter;
        %W_iter = zeros(size(X,1)+1,M^2);
        W_iter = zeros(M, size(X,1)+1, M);
        log_odds_iter = [];
        %[~,ss] = max(gamma); ss is Qest
        clear Atemp;
        Atemp = permute(A, [3 2 1]);
        Anew = zeros(M, M-1, T);
        for t=1:T
            Anew(:,:,t) = reshape(Atemp(t,~eye(M)), M-1, [])';
        end
        for nb=1:M
            %AQUI SECALHAR NAO PASSAR OS ELEMENTOS DIAGONAIS DA MATRIZ PARA
            %A ESTIMAÇAO DOS PESOS -> usei Anew
            %E DEPOIS AO CONCATENAR OS PESOS
            %ADICIONAR OS PESOS KK COMO VETORES DE ZEROS, PORQUE NUNCA
            %SERAO USADOS !! -> done
           
            aux = Anew(nb:M:end,:,:); %A(nb:M:end,:,:);
            %find indexes of the transitions from 1->2, 1->3 if nb=1, 2->1
            %and 2->3 if nb=2 and 3->1 and 3->2 if nb=3
            idxs_acum = [];
            for jj=1:M
                if jj~=nb
                    idxs_acum = [idxs_acum; strfind(Qest', [nb jj])'];
                end
            end
            idxs_acum = sort(idxs_acum);
            %only transitions
            tic
            [logit_model, ~] = logitMn(X(:,idxs_acum), reshape(aux(:,:,idxs_acum), M-1, length(idxs_acum)), 50); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb)));
            toc
            
            %all       
            %[logit_model, ~] = logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M-1, sum(Qest==nb))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb)));

            %[logit_model, ~] = logitMn(X, reshape(At(nb:k:end,:,:), k, n));
            %[~, log_odds] = logitMnPred(logit_model, X);
            %[~, log_odds] = logitMnPred(logit_model, X(:,ss==nb));
            %aux_A(nb, setdiff(1:M,nb),:)=log_odds;
            
            %log_odds_iter = [log_odds_iter; log_odds];        

            
            clear W;
            W = logit_model.W;
            %model.W contains the betas but the last row is the intercept, we will
            %change to put it as the first row
            
            %W = [W(end,:);W];
            %W = W(1:end-1,:);

            %W_iter = [W_iter, W];
            %final_W=[final_W, W];
            %columns = get_columns_(M, nb); %does not work for more than 9 states
            %columns = columns(columns~=get_columns_(M,nb,nb));
            
            %W_iter(:,columns) = W;
            
            W_iter(nb, :, setdiff(1:M,nb)) = W;
        end
%         %% SUGESTÃO DA ZITA: MSE DÁ MAIOR
%         [logit_model, ~] = logitMn(X, reshape(permute(Anew, [2,1,3]), (M-1)*M, length(X)));
%         final_columns = [];
%         for i=1:M
%             columns = get_columns_(M, i);
%             columns = columns(columns~=get_columns_(M,i,i));
%             final_columns = [final_columns, columns];
%         end
%         W_iter = zeros(size(X,1)+1,M^2);
%         W_iter(:,final_columns) = logit_model.W;
%         % SCALING NÃO SEI SE POSSO FAZER ISTO
%         %W_iter = W_iter./sum(sum(W_iter));
%         
%         final_W = [final_W, W_iter];
        %%
 
        % Re-estimate the transition matrix with the estimated weights
       A = zeros(M, M, T);
       
       for i=1:M
           %columns = get_columns_(M, i);
           %columns = columns(columns~=get_columns_(M,i,i));
           for j=setdiff(1:M,i)
               %column = get_columns_(M,i,j);  
               
               A(i,j,:) = exp(W_iter(i,:,j)*[X;ones(1,T)]-logsumexp(reshape(W_iter(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X;ones(1,T)]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
               
               %before I've done this way, but its 20x more time consuming
               %and the results are the same (~10e-15 difference)
               
               %for t=1:T                 
               %    A(i,j,t) = exp(W_iter(i,:,j)*[X(:,t);1]-logsumexp(reshape(W_iter(i,:,setdiff(1:M,i)), size(X,1)+1, M-1)'*[X(:,t);1]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
               %    %A(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter'*[X(:,t);1]));
               %end
           end
       end
       
        %A = aux_A;
        %A=normalize(A,2);
        %if llh(iter)-llh(iter-1) < tol*abs(llh(iter-1)); break; end   % check likelihood for convergence
        if(any(any(round(sum(A,2),1,'significant')~=1))) %check if transition matrix sums to 1, OK: 0, ERROR: 1
            disp('error: transition matrix Logistic regression does not sum to 1!');
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
        
        
        final_W = reshape(final_W, n_features+1,M^2,[]);

        models(ir).A = A;
        models(ir).lambdas = lambdas;
        models(ir).lambdas_tt = lambdas_tt;
        models(ir).s = PAI;
        models(ir).P = P;
        if geometric
            models(ir).PM=PM';
        end
        models(ir).log_odds = log_odds_iter;
        models(ir).Qest = Qest;
        models(ir).B = B;
        
        %change it so that the intercept is first
        %if isempty(W_iter)
        %    W_iter=final_W(:,:,end);
        %end
        %W_iter = [W_iter(:,end,:),W_iter]; %[W_iter(end,:);W_iter];
        %W_iter = W_iter(:, 1:end-1,:); %W_iter(1:end-1,:);
        models(ir).betas = W_iter;
        models(ir).betas_total = final_W;
        
        
    end
    toc
end

if IterationNo>0
    model = models(end);
else
    model.A = A;
    model.lambdas = lambdas;
    model.s = PAI;
    model.P = P;
    model.PM = PM';
    if geometric
        P = PM';
    end
    model.Qest = Qest;
    model.betas = model_betas;
    model.B = B;
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