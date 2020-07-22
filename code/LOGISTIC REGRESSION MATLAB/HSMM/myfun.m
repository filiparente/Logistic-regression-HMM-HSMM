% where myfun is a MATLAB function such as:
% function [f,g] = myfun(x)
% f = sum(sin(x) + 3);
% if ( nargout > 1 ), g = cos(x); end

function [f,g] = myfun(t,x,phi, lambda)
sum_=0;
N = size(t,2);
K = size(t,1);
y =  zeros(N,K);
lny =  zeros(N,K);
%x = reshape(x, K, [])';
n_features = size(x,1);

% for n=1:N
%     for k=1:K
%         y(n,k) = exp(x(:,k)'*phi(:,n));
%     end
%     den = sum(y(n,:));
%     for k=1:K
%         y(n,k) = y(n,k)/den;
%         lny(n,k) = exp(x(:,k)'*phi(:,n)-logsumexp(x'*phi(:,n)));
%         %if ~(y(n,k)>=0 && y(n,k)<=1)
%         %    disp('error');
%         %else if ~(t(n,k)>=0 && t(n,k)<=1)
%         %        disp('error2');
%         %    end
%         %end
%         sum_=sum_+t(n,k)*lny(n,k);
%     end
% end
% %if(sum_>=0)
% %    disp('error3');
% %end
% f=-sum_;
% if (nargout > 1)
%   g = zeros(K,n_features);
%   for j=1:K
%       g(j,:) = sum((y(:,j)-t(:,j))'*phi');
%   end
%   
%   %g = reshape(g,1,[]);
%   %g=g';
% end

%% new
A = x'*phi; 
logY = bsxfun(@minus,A,logsumexp(A,1));            % 4.104
llh = dot(t(:),logY(:))-0.5*lambda*dot(x(:),x(:)); %we want to maximize the likelihood
f = -llh; %minimize this quantity

Y = exp(logY);

g = phi*(Y-t)'+lambda*x;      % 4.96
end
