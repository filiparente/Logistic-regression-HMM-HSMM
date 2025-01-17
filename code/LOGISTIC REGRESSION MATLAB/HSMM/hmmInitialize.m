%% REPLACED 
function [A,B,PAI,Vk,MO,K, lambdas]=hmmInitialize(MO,M,K,MT, pruning, init)
%function [A,B,PM,PAI,Vk,MO,K, lambdas]=hsmmInitialize(MO,M,D,K,MT)
% 
% Author: Shun-Zheng Yu
% Available: http://sist.sysu.edu.cn/~syu/Publications/hsmmInitialize.m.txt
%
% To initialize the matrixes of A,B,P,PAI for hsmm_new.m, to get 
% the observable values and to transform the observations O
% from values to their indexes.
%
% Usage: [A,B,P,PAI,Vk,O,K]=hsmmInitialize(O,M,D,K)
% 
%  N:  Number of observation sequences
%  MT: Lengths of the observation sequences: MT=(T_1,...,T_N)
%  MO: Set of the observation sequences: MO=[O^1,...,O^N], O^n is the n'th obs seq.
%  M:  total number of states
%  K:  total number of observable values
%
%  A: initial values of state transition probability matrix
%  B: initial values of observation probability matrix
%  PAI: initial values of starting state probability vector
%  Vk: the set of observable values
%
%  
%  Updated Nov.2014
%
%

N=size(MO,2);
O=MO(1:MT(1),1);            % a vector of multiple observation sequences
if N>1
    for on=2:N              % for each observation sequence
        T=MT(on);           % the length of the n'th obs seq
        O=[O;MO(1:T,on)];   % the n'th observation sequence
    end
end

[V,I]=sort(O);              % sort the observation V values I corresponding indexes
V=diff([0;V]);              % find the same observable values
Vk=V(V>0);                  % get the set of observable values
Vk=cumsum(Vk);              %same as Vk = unique(V);
%if length(Vk)<K             % compare K with the number of observables
    K=length(Vk);           % let K<=|Vk|
%end
if K<M
    M=K;                    % num of observables must be >= num of states
end
V(V>0)=1;                   % get the cumulative distribution of observables
V=cumsum(V);
V=floor(V./max(V).*K)+1;    % divide the observables into K periods
%K=K+1;
% Associate observations with states
KM=ceil(length(Vk)/double(M));
Vk(KM*M)=Vk(end);
%K=K+1;
Vk1=zeros(KM,M);
Vk1(:)=Vk(1:KM*M);

%% POISSON
%lambdas = sort(rand*mean(Vk1));
if strcmp(init,'smart')
    lambdas = sort(mean(Vk1)); %inicialização "inteligente": lambdas espaçados
elseif strcmp(init,'rand')
    lambdas = sort(rand(1,M)*max(Vk)); %inicialização "inteligente": lambdas espaçados
else
    disp('Mode of initialization of Poisson parameters not recognized.');
    return
end
B = zeros(M, length(Vk));
 for i=1:M
     %ss(i)=sum(log(1:i));
     B(i,:) = poisspdf(Vk, lambdas(i))';
     B(i,:)=B(i,:)/sum(B(i,:));
 end
 %B=exp(log(lambdas')*(1:K)-ones(M,1)*ss-lambdas'*ones(1,K));
 %B=B+(max(B')'.*0.01)*ones(1,K);
 %B=B./(sum(B')'*ones(1,K));      %observation probability distribution matrix

%% MULTINOMIAL 
%B=generate_random_dur_dist(M, K);
%Map the observations to indices 
O(I)=V; 
MO(1:MT(1),1)=O(1:MT(1));
O(1:MT(1))=[];
if N>1
    for on=2:N                  % for each observation sequence
        T=MT(on);               % the length of the n'th obs seq
        MO(1:T,on)=O(1:T);      % the n'th observation sequence
        O(1:T)=[];
    end
end

% other initial probabilities
%    PAI=rand(M,1)+1;
%PAI=ones(M,1);
%PAI=PAI./sum(PAI);              %starting state probabilities
PAI = generate_random_initial_dist(M);

% A=ones(M);
%   A=rand(M)+1;
%    for i=1:M
%       A(i,i)=0;
%    end
% A=A./(sum(A')'*ones(1,M));      %Transition Probability Matrix of states
A = zeros(M, M,MT(1));
for t=1:MT(1)
    A(:,:,t) = generate_random_transition_matrix(M);
end


end
