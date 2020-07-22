function S = markovMySample(model, len, nsamples)
    % Sample from a markov distribution
    % S is of size nsamples-by-len
    %
    
    % This file is from pmtk3.googlecode.com
    
    if nargin < 3, nsamples = 1; end
    pi = model.PAI;
    A = model.A;
    % REPLACED
    P = model.P;
    %PM=model.PM;
    
    S = zeros(nsamples, len);
    
    for ii=1:nsamples
        S(ii, 1) = sampleDiscrete(pi);
        %% REPLACED 
        duration = sampleDiscrete(P(S(ii,1),:));  %pick up the duration for init state BY
        %u = rand;
        
        %duration = ceil(log(1-u) / log(1-PM(S(ii,1)))); %poissrnd(PM(S(ii,1))) %geornd(PM(S(ii,1)))
        
        S(ii, 1:duration) = S(ii, 1);
        tt=duration+1;
        while tt<=len
            S(ii, tt) = sampleDiscrete(A(S(ii, tt-1), :));
            %% REPLACED 
            duration = sampleDiscrete(P(S(ii,tt),:));
            %u = rand;
            %duration = ceil(log(1-u) / log(1-PM(S(ii,tt)))); %poissrnd(PM(S(ii,tt))) %geornd(PM(S(ii,tt)))
            
            S(ii, tt:min(tt+duration-1, len)) = S(ii, tt);
            tt=tt+duration;
        end
        
    end
end
