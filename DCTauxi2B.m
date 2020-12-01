%% Code PHZ_03/09_001.009 V1 F
%% --------------------------------------------------------------------------
%UnWrap data for Cosine Transform
%D: March 09, LU: March 09

%Manual
%Reads   f: function, Y: from fft of y, N:number of points

%Comment: Uses Page 624 Press's Efficent Algorithm 
%         Is used for Real inout data f

%Warning: DOES NOT work fine for complex f
%%  --------------------------------------------------------------------------

function [F]=DCTauxi2B(Y,f,N)


%Find the real and imag parts of Y
R=real(Y);
I=imag(Y);


%This one is not prodeuced naturally (Eq. 12.4.15 of Press)
j=2:N;
F(2)=0.5*(f(1)-f(N+1))+sum(f(j).*cos((j-1)*pi/N));


%This one is not also prodeuced naturally
j=2:N;
F(N+1)=0.5*(f(1)+(-1)^N*f(N+1))+sum(f(j).*(-1).^(j-1));


%Finding the odd terms, (Eq. 12.4.14 of Press)
j=1:N/2;
F(2*j-1)=R(j);


%Finding the even terms
% Do not vectorize this...
for j=2:N/2
    F(2*j)=F(2*(j-1))-I(j);
end
