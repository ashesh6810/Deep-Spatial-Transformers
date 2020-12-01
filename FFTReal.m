%% Code PHZ_03/09_001.005 V1 F
%% --------------------------------------------------------------------------
%Do Efficient forward FFT for Real to Comlex Domain
%D: March 09, LU: March 09

%Manual
%Reads N:number of points, C: control for the point distribution

%Comment: -Works based on Press's algorithm Page 618, modified for Matlab
%         -Modifications indlude changing the indexing order
%          and VERY IMPORTANT: the - sign in exp() becasue of the 
%          difference in definition
%         -Y represents F here
                   
%Warning: Not Yet
%% --------------------------------------------------------------------------

function [Y]=FFTReal(y,N)


%Define the auxillary function h
k=1:N/2;
h(k)=y(2*k-1)+i*y(2*k);  %Eq. 1 


%Use FFT to find H
H=fft(h);


%Use symmetry to find H(N/2+1), page 619 Press
H(N/2+1)=H(1);


%Unwrap H to find Y, Press's Eq. 12.3.6
k=1:N/2+1;
Y(k)=0.5*(H(k)+conj(H(N/2-(k-2))))-i*0.5*(H(k)-...
     conj(H(N/2-(k-2)))).*exp(-2.*pi*i*(k-1)/N);    


% Find the rest of Y
m=N/2+2:N;
Y(m)=conj(Y(N-(m-2)));

