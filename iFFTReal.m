%Code PHZ_03/09_001.013 V1 F
%--------------------------------------------------------------------------
%Do Efficient inverse FFT for Real to Comlex Domain
%D: March 09, LU: March 09

%Manual
%Reads N:number of points, C: control for the point distribution

%Comment: -Works based on Press's algorithm Page 618, modified for Matlab
%         -Modifications indlude changing the indexing order
%          and VERY IMPORTANT: the - sign in exp() becasue of the 
%          difference in definition
%         -Y represents F here
                   
%Warning: Not Yet
%--------------------------------------------------------------------------

function [y]=iFFTReal(Y,N)


%Define the auxillary function H
n=1:N/2;
H(n)=0.5*(Y(n)+conj(Y(N/2-(n-2))))+...
     i*0.5*exp(2.*pi*i*(n-1)/N).*(Y(n)-conj(Y(N/2-(n-2))));


%Use iFFT to find h
h=ifft(H);


%Unwrap h to find y,
y(2*n-1)=real(h(n));
y(2*n)=imag(h(n));   


