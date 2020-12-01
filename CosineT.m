%% Code PHZ_06/01_001.005 V1 F 
%THIS IS THE CORRECT CODE FOR COSINE TRANSFORM
%% --------------------------------------------------------------------------
%D: June 09, LU: June 09 

%Does Real Cosine transform in 0-pi

%f: physical space
%F: Chebychev space

%% --------------------------------------------------------------------------

function [F]=CosineT(f,N)


y=DCTauxiB(f,N);
Y=FFTReal(y,N);
F=DCTauxi2B(Y,f,N);

F(1)=F(1)/N;
F(2:N)=F(2:N)*(2/N);
F(N+1)=F(N+1)/N;


