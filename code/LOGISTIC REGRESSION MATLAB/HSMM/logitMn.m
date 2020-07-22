% code from : https://github.com/PRML/PRMLT/blob/master/chapter04/logitMn.m
function [model, llh] = logitMn(X, t, lambda, weight_init)
    % Multinomial regression for multiclass problem (Multinomial likelihood)
    % Input:
    %   X: d x n data matrix
    % d features, n samples and k classes: model.W is a (d+1) x k weight matrix 
    % (+1 because of the intercept)
    %   t: 1 x n label (1~k)
    %   lambda: regularization parameter
    % Output:
    %   model: trained model structure
    %   llh: loglikelihood
    % Written by Mo Chen (sth4nth@gmail.com).
    if nargin < 3
        lambda = 1e-4;
    end
    X = [X; ones(1,size(X,2))];
    
    %if ~exist('comparision_newton_lbfgs.txt')
    %    fileID = fopen('comparision_newton_lbfgs.txt','w');
    %    fprintf(fileID,'%s %s %s\n','Algorithm','Time','Minimum found');
    %else
    %    fileID = fopen('comparision_newton_lbfgs.txt','a');
    %end
    
    %tic
    %[W, llh1] = newtonBlock(X, t, lambda);
    %elapsed_time1 = toc;
    %model.W=W;
    %model.llh=llh1;
    
    %fprintf(fileID,'%s %s %s\n','Newton Block',num2str(elapsed_time1),num2str(llh1(end)));
    
    tic
    [W, llh] = LBFGS(X, t, lambda, weight_init);
    elapsed_time2 = toc;
    
    %fprintf(fileID,'%s %s %s\n','LBFGS',num2str(elapsed_time2),num2str(-llh));
    
    %disp(['Newton block took ' num2str(elapsed_time1) ' and lbfgs took ' num2str(elapsed_time2) ' seconds.']);
    
    %if(llh>abs(llh1(end)))
    %    disp('likelihood of newton block better than likelihood of lbfgs');
    %end
    
    %tic
    %[W, llh] = newtonRaphson(X, t, lambda);
    %elapsed_time3 = toc;
    
    %fprintf(fileID,'%s %s %s\n','Newton Raphson',num2str(elapsed_time3),num2str(llh(end)));
    
    
    model.W = W;
    model.llh=llh;
    
    %fclose(fileID);
end

%% LFBGS
function [W, llh] = LBFGS(X, t, lambda, weight_init)
    [d,n] = size(X);
    if size(t,1)==1
        %k = max(t);
        %T = sparse(t,1:n,1,k,n,n);
        k=1;
        T=sparse(t);
    else
        k = size(t,1);
        T = t; %sparse(t);
    end
    x0 = weight_init;%zeros(d,k);%randn(d,k);%(1-(-1)).*rand(d*k,1) + (-1);% randn(d*k,1); %zeros 

    %Wolfe
    options = struct('GradObj','on','Display','off','LargeScale','on','HessUpdate','lbfgs', 'StoreN', 10, 'TolX', 1e-4, 'TolFun', 1e-3, 'MaxIter', 100, 'GoalsExactAchieve', 0);
    
    %GradObj : Set to 'on' if gradient available otherwise finited difference
    %				is used.
    
    %Display: Level of display. 'off' displays no output; 'plot' displays
    %				all linesearch results in figures. 'iter' displays output at  each 
    %               iteration; 'final' displays just the final output; 'notify' 
    %				displays output only if the function does not converge; 
    
    %HessUpdate : If set to 'bfgs', Broyden?letcher?oldfarb?hanno 
    %				optimization is used (default), when the number of unknowns is 
    %				larger then 3000 the function will switch to Limited memory BFGS, 
    %				or if you set it to 'lbfgs'. When set to 'steepdesc', steepest 
    %				decent optimization is used.
    
    %StoreN : Number of itterations used to approximate the Hessian,
    %			 	in L-BFGS, 20 is default. A lower value may work better with
    %				non smooth functions, because than the Hessian is only valid for
    %				a specific position.
    
    %TolX : Termination tolerance on x, default 1e-6.
    
    %TolFun : Termination tolerance on the function value, default 1e-6.
    
    %MaxIter : Maximum number of iterations allowed, default 400.
    
    %GoalsExactAchieve : If set to 0, a line search method is
    %               used which uses a few function calls to do a good line
    %               search. When set to 1 a normal line search method with Wolfe 
    %				conditions is used (default).
    tic	
    func = @(x) myfun(T,x,X, lambda);
    [x,fval2,exitflag,output,grad] = fminlbfgs(func,x0,options);
    W = reshape(x,d,k);
    llh = fval2;
    
    
end

function [W, llh] = BOA(X, t, lambda)
    [d,n] = size(X);
    if size(t,1)==1
        %k = max(t);
        %T = sparse(t,1:n,1,k,n,n);
        k=1;
        T=sparse(t);
    else
        k = size(t,1);
        T = t; %sparse(t);
    end

    %B = 2*kron((eye(k)+ones(k,1)*ones(k,1)'),pinv(X*X'));
    %B = 2*kron((eye(k)+ones(k,1)*ones(k,1)'),inv(X'*X));
    %aux_B = 0.5*kron((eye(k)-(ones(k,1)*ones(k,1)')/(k+1)),eye(d));
    
end
function [W, llh] = newtonRaphson(X, t, lambda)
    [d,n] = size(X);
    if size(t,1)==1
        %k = max(t);
        %T = sparse(t,1:n,1,k,n,n);
        k=1;
        T=sparse(t);
    else
        k = size(t,1);
        T = t; %sparse(t);
    end
    tol = 1e-3; %-4
    tol2=1e-1;
    maxiter = 100;
    llh = -inf(1,maxiter);
    max_elapsediter = 5;
    elapsediter = 0;
    max_llh = -inf;
    W = zeros(d,k);
    
    dk = d*k;
    idx = (1:dk)';
    dg = sub2ind([dk,dk],idx,idx);
    
    HT = zeros(d,k,d,k);
    %prev_G = zeros(d,k);
    %n = size(X,2);
 
    for iter = 2:maxiter
        A = W'*X;                                        % 4.105
        logY = bsxfun(@minus,A,logsumexp(A,1));            % 4.104
        llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));  % 4.108
        if llh(iter)>max_llh
            if abs(llh(iter)-max_llh)<tol2
                break;
            else
                max_llh = llh(iter);
                max_iter = iter;
                W_iter = W;
                elapsediter = 0;
            end
        end
        if (abs(llh(iter)-llh(iter-1)) < tol || elapsediter > max_elapsediter); break; end
        elapsediter = elapsediter+1;
        
        Y = exp(logY);
        for i = 1:k
             for j = 1:k
                r = Y(i,:).*((i==j)-Y(j,:));  % r has negative value, so cannot use sqrt
                HT(:,i,:,j) = bsxfun(@times,X,r)*X';     % 4.110
            end
        end
        
        %% Gradient OPTION 1: original   
        G = X*(Y-T)'+lambda*W;      % 4.96
       
        
        %% Gradient OPTION 2: kron -> the same
        %G=zeros(dk,1);
        %for i=1:size(X,2)
        %    G = G+kron(Y(:,i)-T(:,i),X(:,i));
        %end
        %G = reshape(G, d, k);
        %G2 = G+lambda*W;
        
        %if ~all(all(abs(G1-G2)<1e-10))
        %    disp('error');
        %end
        
        %% Hessian OPTION 1: original      
        H = reshape(HT,dk,dk);
        H(dg) = H(dg)+lambda;

        %% Weight update OPTION 1: original
        W(:) = W(:)-H\G(:);   

        %% Weight update OPTION 2: BOA 1     
        %W(:) = W(:)+B*G(:);   
        
        %% Weight update OPTION 3: BOA 2
        %W(:) = B*G(:)+B*(aux_B*W(:));    

        %% OPTION 4: approximate inverse Hessian
        %[new_Hessian, ~] = approx_invHess(-H*G(:), prev_G, G, H);
        %W(:) = W(:)-new_Hessian*G(:);           
        %prev_G = G;
         
     end
    llh = llh(2:iter);
    if max_llh>llh(end)
        W=W_iter;
        llh = max_llh;
        max_iter
    else
        iter
    end
end

function [W, llh] = newtonBlock(X, t, lambda)
    [d,n] = size(X);
    if size(t,1)==1
        %k = max(t);
        %T = sparse(t,1:n,1,k,n,n);
        k=1;
        T=sparse(t);
    else
        k = size(t,1);
        T = t; %sparse(t);
    end
    %k = max(t);
    idx = (1:d)';
    dg = sub2ind([d,d],idx,idx);
    tol = 1e-4; %-3
    tol2=1e-1;
    maxiter = 100;
    max_elapsediter = 50; %20
    elapsediter = 0;
    llh = -inf(1,maxiter);
    max_llh = -inf;

    %T = sparse(t,1:n,1,k,n,n);
    W = zeros(d,k);
    A = W'*X;
    logY = bsxfun(@minus,A,logsumexp(A,1));
    for iter = 2:maxiter
        for j = 1:k
            Y = exp(logY);
            %den = sum(sum(T.*Y));
            Xw =  bsxfun(@times,X,sqrt(Y(j,:).*(1-Y(j,:))));
            %Xw = bsxfun(@times,X,(2.*Y(j,:).^2.*(1-Y(j,:))-Y(j,:).*(1-Y(j,:)).*T(j,:))/(den^2));
            H = Xw*Xw';
            H(dg) = H(dg)+lambda;
            g = X*(Y(j,:)-T(j,:))'+lambda*W(:,j);
            %g = X*(Y(j,:).*Y(j,:)-T(j,:))'/den +lambda*W(:,j);
            W(:,j) = W(:,j)-H\g;
            A(j,:) = W(:,j)'*X;
            logY = bsxfun(@minus,A,logsumexp(A,1));  % must be here to renormalize
        end
        llh(iter) = dot(T(:),logY(:))-0.5*lambda*dot(W(:),W(:));
        if llh(iter)>max_llh
            if abs(llh(iter)-max_llh)<tol2
                break;
            else
                max_llh = llh(iter);
                max_iter = iter;
                W_iter = W;
                elapsediter = 0;
            end
        end
        if (abs(llh(iter)-llh(iter-1)) < tol) || elapsediter > max_elapsediter; break; end
        elapsediter = elapsediter+1;
    end
    llh = llh(2:iter);
    
    if max_llh>llh(end)
        W=W_iter;
        llh = max_llh;
        max_iter
    else
        iter
    end
end

%% -----------------------   OTHER CODE  ---------------------------------

%% OLD LBFGS
%l = ones(d*k,1)*-inf;
%u = ones(d*k,1)*inf;
%opts = struct('display',true,'xhistory',true,'max_iters',25);
%func = @(x) logitreg(T,x,X);
%[x,xhistory] = LBFGSB(func,x0,l,u,opts);
%xhist_unc_x = xhistory(1,:);
%xhist_unc_y = xhistory(2,:);
%llh = xhistory(:,2);

%% FMINUNC
% Initialize weights with newton block

%[W, ~] = newtonBlock(X, t, lambda);%W = zeros(d,k);
%func = @(x) logitreg(T,x,X, lambda);  
%options = optimoptions('fminunc','SpecifyObjectiveGradient',true, 'HessianFcn', 'objective', 'Algorithm','trust-region');
%x = fminunc(func,W, options);

%% LOGISTIC REGRESSION
% function [f,g,h] = logitreg(t,x,phi, lambda)
% sum_=0;
% N = size(t,2);
% K = size(t,1);
% y =  zeros(N,K);
% lny =  zeros(N,K);
% x = reshape(x, K, [])';
% n_features = size(x,1);
% HT = zeros(n_features,K,n_features,K);
% %t=t';
% % for n=1:N
% %     for k=1:K
% %         y(n,k) = exp(x(:,k)'*phi(:,n));
% %     end
% %     for k=1:K
% %         y(n,k) = y(n,k)/sum(y(n,:));
% %         lny(n,k) = exp(x(:,k)'*phi(:,n)-logsumexp(x'*phi(:,n)));
% %         if ~(y(n,k)>=0 && y(n,k)<=1)
% %             disp('error');
% %         else if ~(t(n,k)>=0 && t(n,k)<=1)
% %                 disp('error2');
% %             end
% %         end
% %         sum_=sum_+t(n,k)*lny(n,k);%log(y(n,k));
% %     end
% % end
% % if(sum_>=0)
% %     disp('error3');
% % end
% % f=-sum_;
% % if (nargout > 1)
% %   g = zeros(K,n_features);
% %   for j=1:K
% %       g(j,:) = sum((y(:,j)-t(:,j))'*phi');
% %   end
% %   
% %   g = reshape(g,1,[]);
% %   g=g';
% % end
% A = x'*phi; 
% %A=A';% 4.105
% logY = bsxfun(@minus,A,logsumexp(A,1));            % 4.104
% f = -dot(t(:),logY(:))+0.5*lambda*dot(x(:),x(:)); %minimize this quantity
% 
% Y = exp(logY);
% 
% for i = 1:K
%     for j = 1:K
%        r = Y(i,:).*((i==j)-Y(j,:));  % r has negative value, so cannot use sqrt
%        HT(:,i,:,j) = bsxfun(@times,phi,r)*phi';     % 4.110
%    end
% end
% g = phi*(Y-t)'+lambda*x;      % 4.96
%  
% dk = n_features*K;
% h = reshape(HT,dk,dk);
% 
% idx = (1:dk)';
% dg = sub2ind([dk,dk],idx,idx);
%    
% h(dg) = h(dg)+lambda;
% 
% end
% 
% 
% function [new_Hessian, new_dir] = approx_invHess(dir, old_gradient, new_gradient, old_Hessian)
%     alpha=1;
%     deltaX=alpha* dir;
%     deltaG=new_gradient(:)-old_gradient(:);
%     N = size(old_Hessian,1);
% 
%     %if ((deltaX'*deltaG) >= sqrt(eps)*max( eps,norm(deltaX)*norm(deltaG) ))
%         
%         %% Default BFGS as described by Nocedal
%         p_k = 1 / (deltaG'*deltaX);
%         Vk = eye(N) - p_k*deltaG*deltaX';
%         % Set Hessian
%         new_Hessian = Vk'*old_Hessian *Vk + p_k * (deltaX*deltaX');
%         % Set new Direction
%         new_dir = -new_Hessian*new_gradient(:);
%         
%         %% L-BFGS with scaling as described by Nocedal
% 
% %         % Update a list with the history of deltaX and deltaG
% %         data.deltaX(:,2:optim.StoreN)=data.deltaX(:,1:optim.StoreN-1); data.deltaX(:,1)=deltaX;
% %         data.deltaG(:,2:optim.StoreN)=data.deltaG(:,1:optim.StoreN-1); data.deltaG(:,1)=deltaG;
% % 
% %         data.nStored=data.nStored+1; if(data.nStored>optim.StoreN), data.nStored=optim.StoreN; end
% %         % Initialize variables
% %         a=zeros(1,data.nStored);
% %         p=zeros(1,data.nStored);
% %         q = data.gradient;
% %         for i=1:data.nStored
% %             p(i)= 1 / (data.deltaG(:,i)'*data.deltaX(:,i));
% %             a(i) = p(i)* data.deltaX(:,i)' * q;
% %             q = q - a(i) * data.deltaG(:,i);
% %         end
% %         % Scaling of initial Hessian (identity matrix)
% %         p_k = data.deltaG(:,1)'*data.deltaX(:,1) / sum(data.deltaG(:,1).^2); 
% % 
% %         % Make r = - Hessian * gradient
% %         r = p_k * q;
% %         for i=data.nStored:-1:1,
% %             b = p(i) * data.deltaG(:,i)' * r;
% %             r = r + data.deltaX(:,i)*(a(i)-b);
% %         end
% % 
% %         % Set new direction
% %         data.dir = -r;
%         
%     %end
% end

