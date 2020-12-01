%% Code PHZ_03/09_001.008 V1 F
%% --------------------------------------------------------------------------
%Wrap data for Cosine Transform
%D: March 09, LU: March 09

%Manual
%Reads   f: function, N:number of points

%Comment: Uses Page 624 Press's Efficent Algorithm 
%         Is used for Real inout data f

%Warning: Not yet
%% --------------------------------------------------------------------------

function [y]=DCTauxiB(f,N)


%Eq. 12.4.13, the auxiliary function y 
j=1:N;
y(j)=0.5*(f(j)+f(N-(j-2)))-sin((j-1)*pi/N).*(f(j)-f(N-(j-2)));