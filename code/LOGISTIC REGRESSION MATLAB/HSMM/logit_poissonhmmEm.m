function  [model, llh, iter] = logit_poissonhmmEm(params, x, X, max_iterations, tolerance, mode, Vk)
% EM algorithm to fit the parameters of HMM model (a.k.a Baum-Welch algorithm)
% Input:
%   X: n_features x n, feature vector
%   x: 1 x n integer vector which is the sequence of observations
%   params: model or n_states
%   mode - only applicable if the logistic regression is being used. 
%   if true, all features are used to estimate the weights of the
%   logistic regression; if false, only the features of the timestamps
%   associated with a transition, according to the state sequence
%   estimated by the EM algorithm.
% Output:s
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
dim = size(x,2);
N = size(x,1);

%X = sparse(x,1:n,1,d,n);
if isstruct(params)   % params with a model
    At = params.A;
    lambdas = params.lambdas;
    PAI = params.PAI;
    B = params.B;
    k = size(At,1);
elseif numel(params) == 1  % random params with latent k
    k = params;
    %from lambdas vector retrieve k
    %k = numel(params);
    %At = zeros(k,k,n);
    %for t=1:n
    %    At(:,:,t) = normalize(rand(k,k),2);
    %end
    A = normalize(rand(k,k),2);
    At = repmat(A, 1,1,n); %all transition matrix for all t are equal
    % AT POSSO INICIALIZAR ASSIM OU TENHO DE INICIALIZAR OS BETAS E DEPOIS
    % CALCULAR O AT COM AS FEATURES E OS BETAS (EXP/SUM EXP)????????????
    
    %A = normalize(rand(k,k),2);
    %lambdas = params;
    [V,~]=sort(x');              % sort the observation V values I corresponding indexes
    V=diff([0;V]);              % find the same observable values
    Vk=V(V>0);                  % get the set of observable values
    Vk=cumsum(Vk);              %same as Vk = unique(V);
    KM=ceil(length(Vk)/double(k));
    Vk(KM*k)=Vk(end);
    %K=K+1;
    Vk1=zeros(KM,k);
    Vk1(:)=Vk(1:KM*k);
    lambdas = sort(mean(Vk1));
    s = normalize(rand(k,1),1);
end
%M = E*X;
if isfield(params, 'pn')
    newton_param = params.pn;
else
    newton_param = 0.8;
end
M = zeros(k,dim);
trans_pos = zeros(k, k, dim);
final_W = [];
Vk = unique(x);
for i=1:k
    M(i,:) = poisspdf(x,lambdas(i));
    M(i,:) = M(i,:)/(sum(poisspdf(Vk, lambdas(i))));
end
M = M + 1e-100; %to avoid numerical errors (Nan's in gamma)

%check if the normalization is correct
E_ = (pinv(full(sparse(x,1:dim,1,max(x),dim)))'*M')'; %emission matrix
if(round(sum(E_,2),1,'significant')~= ones(k,1))
    disp('ERROR');
end %emission must sum to 1 on the rows!!

%M = normalize(M,1);
llh = -inf(1,max_iterations);
lambdas_tt = {};

for iter = 2:max_iterations
%     E-step
    [At, trans_pos, gamma,alpha,beta,c] = hmmSmoother_At(M,At,PAI);
    llh(iter) = sum(log(c(c>0)));
    
%     M-step 
    lambdas = sum(gamma*x',2)./sum(gamma,2);
    for i=1:k
        M(i,:) = poisspdf(x,lambdas(i));
        M(i,:) = M(i,:)/(sum(poisspdf(Vk, lambdas(i))));
    end 
    M = M + 1e-100; %to avoid numerical errors (Nan's in gamma)

    %check if the normalization is correct
    E_ = (pinv(full(sparse(x,1:dim,1,max(x),dim)))'*M')'; %emission matrix
    if(any(round(sum(E_,2),2,'significant')~= ones(k,1))) %emission matrix
        disp('ERROR');
    end %emission must sum to 1 on the rows!!
    
    PAI = gamma(:,1)/sum(gamma(:,1));%gamma(:,1)
    [Qest, ~] = hmmViterbi_(M, At, PAI);
    Qest = Qest';
    
    At(:,:,end) = At(:,:,end-1);%ones(k,k)./k;
    aux=sum(At,2);
    if(any(round(aux(:),2,'significant')~= ones(k*dim,1))) 
        disp('ERROR');
    end 
    clear aux;
    
    if iter==2
        clear W_iter;
        W_iter = zeros(k, size(X,1)+1, k);
    end
             
    clear Atemp;
    Atemp = permute(At, [2 1 3]);
 
    for nb=1:k
        aux = Atemp(nb:k:end,:,:); %A(nb:M:end,:,:);
        if mode
            %% SUGESTÃO DA ZITA: MSE DÁ MAIOR
           [logit_model, ~] = logitMn(X, reshape(aux, k-1, length(X)), newton_param, squeeze(W_iter(nb, :, setdiff(1:k,nb)))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb));
        else
           %find indexes of the transitions from 1->2, 1->3 if nb=1, 2->1
           %and 2->3 if nb=2 and 3->1 and 3->2 if nb=3
           idxs_acum = [];
           for jj=1:k
              idxs_acum = [idxs_acum; strfind(Qest', [nb jj])'];   
           end
           idxs_acum = sort(idxs_acum);
           %only transitions
           [logit_model, ~] = logitMn(X(:,idxs_acum), reshape(aux(:,:,idxs_acum), k, length(idxs_acum)), newton_param, squeeze(W_iter(nb, :, :))); %logitMn(X(:,Qest==nb), reshape(aux(:,:,Qest==nb), M, sum(Qest==nb)));
        end
        clear W;
        W = logit_model.W;
        
        %logistic regression weights Bii are not used -> zeros
        W_iter(nb, :, :) = W;
    end
 

    % Re-estimate the transition matrix with the estimated weights
     At = zeros(k, k, dim);
  
%     for i=1:k
%         for j=1:k
%              At(i,j,:) = exp(W_iter(i,:,j)*[X;ones(1,dim)]-logsumexp(reshape(W_iter(i,:,:), size(X,1)+1, k)'*[X;ones(1,dim)]));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
%         end
%     end
    
    %diferença <1e-14 e 98% + rapido (102s vs. 2s)
    for i=1:k
        At(i,:,:) = exp(squeeze(W_iter(i,:,:))'*[X;ones(1,dim)]-repmat(logsumexp(reshape(W_iter(i,:,:), size(X,1)+1, k)'*[X;ones(1,dim)]), k, 1));%exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
    end
   
    if(any(any(round(sum(At,2),1,'significant')~=1))) %check if transition matrix sums to 1, OK: 0, ERROR: 1
        disp('error: transition matrix Logistic regression does not sum to 1!');
    end
    

    if iter==8
        disp('aqui');
    end
   if llh(iter)-llh(iter-1) < tolerance*abs(llh(iter-1)); break; end   % check likelihood for convergence
end
%llh = llh(2:iter);
n_features = size(X,1);
final_W = reshape(final_W, n_features+1,k^2,[]);

%model.A = A;
model.A = At;
model.lambdas = lambdas;
model.s = PAI;
model.store_GAMMA = gamma;
%model.log_odds = log_odds_iter;
%change it so that the intercept is first
%W_iter = [W_iter(end,:);W_iter];
%W_iter = W_iter(1:end-1,:);
model.betas = W_iter;
disp(iter);

end

%% OLD CODE
% %At == normalize(trans_pos,2)
% s = gamma(:,1)/sum(gamma(:,1));%gamma(:,1)                                                                             % 13.18
% %M = bsxfun(@times,gamma*X',1./sum(gamma,2))*X;
% 
% %UNSUPERVISED CASE: i do not have access to the state sequence,
% %instead, I have access to the state posteriors M
% %k newton blocks
% clear W_iter;
% W_iter = [];
% log_odds_iter = [];
% [~,ss] = max(gamma);
% 
% %log_odds_iter = zeros(k^2, n);
% for nb=1:k
%     aux = At(nb:k:end,:,:);
%     [logit_model, ~] = logitMn(X(:,ss==nb), reshape(aux(:,:,ss==nb), k, sum(ss==nb)));
%     logit_model.W
% 
%     %[logit_model, ~] = logitMn(X, reshape(At(nb:k:end,:,:), k, n));
%     [~, log_odds] = logitMnPred(logit_model, X);
%     %[~, log_odds] = logitMnPred(logit_model, X(:,ss==nb));
% 
%     log_odds_iter = [log_odds_iter; log_odds];        
% 
%     %model.W contains the betas but the last row is the intercept, we will
%     %change to put it as the first row
%     clear W;
%     W = logit_model.W;
%     %W = [W(end,:);W];
%     %W = W(1:end-1,:);
%     W_iter = [W_iter, W];
%     final_W=[final_W, W];
% 
% %         for nn=1:k
% %             clear idxs;
% %             idxs = strfind(ss,[nb nn]);
% %             aux = At(nb, nn, idxs);
% %             [logit_model, ~] = logitMn(X(:,idxs), aux(:)');
% %             logit_model.W
% %             W_iter = [W_iter, logit_model.W];
% %         end
% end
% %logit_model.W = W_iter;
% %X = X';
% %[y, P] = logitMnPred(logit_model, X); %y is the predicted state sequence and P is the state posteriors according to the logistic regression model
% 
% %Att = zeros(k,k,n);
% %Att = permute(reshape(P, k, k, n), [k k-1 k+1]);
% %Att = normalize(Att, 2);
% %W_iter = W_iter./sum(sum(W_iter));
% for t=1:n
%     for i=1:k
%         for j=1:k
%             column = get_columns(k,i,j);
%             columns = get_columns(k, i);
%             At(i,j,t) = exp(W_iter(:,column)'*[X(:,t);1])/sum(exp(W_iter(:,columns)'*[X(:,t);1]));
%         end    
%     end
% end
