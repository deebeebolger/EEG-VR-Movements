function [vel,trs] = velocity_calc(dataIn,T)

% [Y,Ty] = resample(X,Tx,Fs,P,Q) interpolates X to an intermediate
%     uniform grid with sample rate equal Q*Fs/P and filters the result
%     using UPFIRDN to upsample by P and downsample by Q.  Specify P and Q
%     so that Q*Fs/P is least twice as large as the highest frequency in the
%     input signal.
%      2*N*max(1,Q/P)

[datax_rs,~] = resample(dataIn(:,1),T,8,10,1);   %(p/q)/fs
[datay_rs,~] = resample(dataIn(:,2),T,8,10,1);
[dataz_rs,trs] = resample(dataIn(:,3),T,8,10,1);

dataIn_rs = [datax_rs,datay_rs,dataz_rs];

D = squareform(pdist(dataIn_rs,'euclidean'));    %Calculate the euclidean distance
Distances = diag(D,1);                        %Find the first super diagonal
vel = Distances./diff(trs);                     %Calculate the velocity

end 