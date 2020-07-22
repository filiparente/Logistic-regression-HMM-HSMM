%% Copyright(c) Naushad Ansari, 2017.
% %% Please feel free to use this open-source code for research purposes only. 
% %%
% %% contact at naushadansari09797@gmail.com in case of any query.
% %%
% %%
% %% This function calculates the mae of a signal with reference to original 
% signal. mae can be calculated for 1-D/2-D/3-D signals.
%%-----------------------------------------------------------------------%%
%%-----------------------------------------------------------------------%%
% %% output: mae-> mae (mean absolute error)
%            
% %% input:  orgSig-> original 1-D/2-D/3-D signal (or reference signal)
%            recSig-> reconstructed (1-D/2-D/3-D) signal/ signal obtained 
%            from the experiment/ signal, of which mae is to be calculated 
%            with reference to original signal.
%            boun-> boun is the boundary left at the corners for the 
%            mae calculation.  default value = 0
%%-----------------------------------------------------------------------%%
%%-----------------------------------------------------------------------%%
function mae = calMAE(orgSig,recSig,varargin)
if isempty(varargin)
    boun = 0;
else boun = varargin{1};
end
if size(orgSig,2)==1       % if signal is 1-D
    orgSig = orgSig(boun+1:end-boun,:);
    recSig = recSig(boun+1:end-boun,:);
else                       % if signal is 2-D or 3-D
    orgSig = orgSig(boun+1:end-boun,boun+1:end-boun,:);
    recSig = recSig(boun+1:end-boun,boun+1:end-boun,:);
end
absErr = norm(orgSig(:)-recSig(:),1);
mae = absErr/length(orgSig(:));