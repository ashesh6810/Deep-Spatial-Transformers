clc
close all
% h=figure(1);
% %%
tau_f = 15.;
tau_d = 100.;
nu    = 0.01;
beta  = 0.196;
sigma = 3.5;
rng('shuffle')
tf=81000;
p = 8.0;

% F = struct('cdata',[],'colormap',[]);
% writerObj = VideoWriter('B3xR2','Uncompressed AVI');
% writerObj.FrameRate =1;
% open(writerObj)


Lx = 46;
Ly = 23;
M=40;
N  = M*30000;
dt = 1./M;
toc11=0;

nx = 128;
ny = 192;

%%
nux = 8.0*((Lx/nx/pi)^p)/dt/8.0;
nux
nuy = 8.0*((1./ny)^p)/dt/8.0;
nuy

x = Lx*(0:nx-1)/nx;
kx = [0:nx/2 -nx/2+1:-1]*2*pi/Lx;

[D,y]=cheb(ny);
y=y*Ly;
D=D/Ly;
D2=D*D;
I=eye(ny+1,ny+1);
PSIR=zeros(2,nx,ny+1);
psiR=zeros(2,nx,ny+1);
for j=1:ny+1
    PSIR(:,:,j) = -sigma*tanh((y(j))/sigma);
    psiR(1,:,j)=fft(squeeze(PSIR(1,:,j)));
    psiR(2,:,j)=fft(squeeze(PSIR(2,:,j)));
end
psiR(:,nx/2+1,:)=0.0;

m=0;
PSIO=zeros(2,nx,ny+1);
psi=zeros(2,nx,ny+1);
for i=1:nx
    for j=1:ny+1
        PSIO(1,i,j) = 1e-2*sin(5*2*pi*x(i)/Lx)*exp(-y(j)^2/(2*sigma^2))+1e-8*randn*exp(-y(j)^2/(1.*sigma^2));
        PSIO(2,i,j) = 1e-3*sin(4*2*pi*x(i)/Lx)*exp(-y(j)^2/(2*sigma^2))+1e-9*randn*exp(-y(j)^2/(1.*sigma^2));        
    end
end

PSIO(1,:,:) = squeeze(PSIR(1,:,:)+PSIO(1,:,:));

% pause

for j=1:ny+1
    psi(1,:,j) = fft(squeeze(PSIO(1,:,j)));
    psi(2,:,j) = fft(squeeze(PSIO(2,:,j)));
end
psi(:,nx/2+1,:)=0.0;
qpvo=zeros(2,nx,ny+1);
for i=1:nx
    qpvo(1,i,:)=(D2-(kx(i))^2*I)*squeeze(psi(1,i,:))-squeeze(psi(1,i,:)-psi(2,i,:));
    qpvo(2,i,:)=(D2-(kx(i))^2*I)*squeeze(psi(2,i,:))+squeeze(psi(1,i,:)-psi(2,i,:));
end
qpvo(:,nx/2+1,:)=0.0;

%%
dx_psi=zeros(2,nx,ny+1);
dx_qpv=zeros(2,nx,ny+1);
dy_psi=zeros(2,nx,ny+1);
dy_qpv=zeros(2,nx,ny+1);
nn=zeros(2,nx,ny+1);
QPV=zeros(2,nx,ny+1,N/M,'single');
PSI=zeros(2,nx,ny+1,N/M,'single');
dx_PSI=zeros(2,nx,ny+1);
dy_PSI=zeros(2,nx,ny+1);
dx_QPV=zeros(2,nx,ny+1);
dy_QPV=zeros(2,nx,ny+1);
m1=1;

% filename = sprintf('%s-%1.3f.mat','Results-tf',tau_f);

% filename = sprintf('%s.mat','Results');
% if (exist(filename)==2)
%filename = sprintf('%s%d.mat','Results',ebi);
%load(filename,'Aold','qpvo');
% end
%filename = sprintf('%s%d.mat','Results',ebi+1);

tic
for n=m1:N
    psi = PSICalc(qpvo,psiR,D2,kx);
    
      for i=1:nx
        dx_psi(:,i,:)=1i*kx(i)*psi(:,i,:);
        dx_qpv(:,i,:)=1i*kx(i)*qpvo(:,i,:);
      end
        dy_psi(1,:,:)=(D*(squeeze(psi(1,:,:)))')';
        dy_psi(2,:,:)=(D*(squeeze(psi(2,:,:)))')';
        dy_qpv(1,:,:)=(D*(squeeze(qpvo(1,:,:)))')';
        dy_qpv(2,:,:)=(D*(squeeze(qpvo(2,:,:)))')';
    dx_psi(:,nx/2+1,:)=0.0;
    dy_psi(:,nx/2+1,:)=0.0;
    dx_qpv(:,nx/2+1,:)=0.0;
    dy_qpv(:,nx/2+1,:)=0.0;
    
    %for j=1:ny+1
        dx_PSI = real(ifft(dx_psi,[],2));
        dx_QPV= real(ifft(dx_qpv,[],2));
        dy_PSI = real(ifft(dy_psi,[],2));
        dy_QPV = real(ifft(dy_qpv,[],2));
    %end
    
    NN = (dx_PSI.*(dy_QPV+beta)-dy_PSI.*dx_QPV);
%     for j=1:ny+1
        nn=fft(NN,[],2);
%     end
    nn(:,nx/2+1,:)=0.0;
    Anew=-nn;
    for i=1:nx
        Anew(1,i,:)=squeeze(Anew(1,i,:)+(psi(1,i,:)-psi(2,i,:)-psiR(1,i,:))/tau_d);
        Anew(2,i,:)=squeeze(Anew(2,i,:)-(psi(1,i,:)-psi(2,i,:)-psiR(2,i,:))/tau_d)-(D2-(kx(i))^2*I)*squeeze(psi(2,i,:))/tau_f;
    end
    Anew(:,nx/2+1,:)=0.0;
    
    if(n==1)
        qpvn=qpvo+dt*Anew;
    else
        qpvn=qpvo+0.5*dt*(3.*Anew-Aold);
    end
    Aold=Anew;
    qpvn(:,nx/2+1,:)=0.0;
    
    qpvo = HyperVis(qpvn,kx,nux,nuy,p,dt,n);
    
    if(mod(n,M)==0)
%         close(h)
        m=m+1;
        disp(m)        
        psi = PSICalc(qpvo,psiR,D2,kx);
%         for j=1:ny+1
            QPV(:,:,:,m) = real(ifft(qpvo,[],2));
            PSI(:,:,:,m)= real(ifft(psi,[],2));
%         end
%        for i=1:nx
%           U(1,i,:,m)=-D*squeeze(PSI(1,i,:,m)); 
%           U(2,i,:,m)=-D*squeeze(PSI(2,i,:,m));
%        end
        CFL = dt*max(max(max(abs(dy_PSI(:,:,:)))))/(Lx/(nx))
        
%         h=figure('OuterPosition',[-5.5 34.5 1293 693]);
%          subplot(2,1,1)
%         contour(x,y(10:end-9),squeeze(PSI(1,:,10:end-9,m))',10);colorbar
%         title(['PSI top, Model Day' num2str(n/(M))])
%         subplot(2,1,2)
%         plot(squeeze(mean(U(1,:,:,m),2)),y,'LineWidth',2);hold on; scatter(squeeze(mean(U(1,:,:,m),2))',y,20,'square','filled','r');
%         title(['U top, Model Day' num2str(n/(M))])
%         drawnow
%         F(n) = getframe(h);
%         a=F(n).cdata;
%         writeVideo(writerObj, a)
        
        if isnan(CFL)
           fid = fopen( 'CFL PROBLEM.txt', 'wt' );
           fclose(fid);
%            close(writerObj)
           break
        end
        toc
        toc11=toc11+toc;
          if toc11>tf
           save(filename,'QPV','PSI','x','y','Aold','qpvo','m','-v7.3')
           toc11=0;
          end
        tic
    end 
end
% close(writerObj)
if n==N 
fid = fopen( 'done.txt', 'wt' );
fprintf(fid,'done');
fclose(fid);
save(filename,'QPV','PSI','x','y','Aold','qpvo','m','-v7.3')
else 
save(filename,'QPV','PSI','x','y','Aold','qpvo','m','-v7.3')
end


