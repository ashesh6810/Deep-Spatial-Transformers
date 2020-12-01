function psi = PSICalc(qpvo,psiR,D2,kx)

nx = size(qpvo,2);
ny = size(qpvo,3)-1;
psi=zeros(2,nx,ny+1);
I = eye(ny+1,ny+1);

for i=1:nx
    RHS = squeeze(qpvo(1,i,:)+qpvo(2,i,:));
    if(i==1)
        RHS(1,1)=psiR(1,1,1);
        RHS(end,1)=psiR(1,1,end);
        A = D2-(kx(i))^2*I;
        A(1,:)=0.0;
        A(1,1)=1.0;
        A(end,:)=0.0;
        A(end,end)=1.0;
        phi1=A\RHS;
    else
        RHS(1,1)=0.0;
        RHS(end,1)=0.0;
        A = D2-(kx(i))^2*I;
        A(1,:)=0.0;
        A(1,1)=1.0;
        A(end,:)=0.0;
        A(end,end)=1.0;
        phi1=A\RHS;
    end
    RHS = squeeze(qpvo(1,i,:)-qpvo(2,i,:));
    if(i==1)
        RHS(1,1)=psiR(1,1,1);
        RHS(end,1)=psiR(1,1,end);
        A = D2-(2.0+(kx(i))^2)*I;
        A(1,:)=0.0;
        A(1,1)=1.0;
        A(end,:)=0.0;
        A(end,end)=1.0;
        phi2=A\RHS;
    else
        RHS(1,1)=0.0;
        RHS(end,1)=0.0;
        A = D2-(2.0+(kx(i))^2)*I;
        A(1,:)=0.0;
        A(1,1)=1.0;
        A(end,:)=0.0;
        A(end,end)=1.0;
        phi2=A\RHS;
    end
    psi(1,i,:)=(phi1+phi2)/2.0;
    psi(2,i,:)=(phi1-phi2)/2.0;
end