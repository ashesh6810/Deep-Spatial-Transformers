%% Code PHZ_06/01_001.008 V1 F 
%THIS IS THE CORRECT CODE FOR inverse COSINE TRANSFORM
%% --------------------------------------------------------------------------
%D: June 09, LU: June 09 

%Does Realinverse Cosine transform in 0-pi

%F: physical space
%f: Chebychev space

%% --------------------------------------------------------------------------

function [f]=iCosineT(F,N)

F(1)=2.*F(1);
F(N+1)=2.*F(N+1);

y=DCTauxiB(F,N);
Y=FFTReal(y,N);
f=DCTauxi2B(Y,F,N);