function qpvo = HyperVis(qpvn,kx,nux,nuy,p,dt,n)

nx = size(qpvn,2);
ny = size(qpvn,3)-1;

Q=zeros(2,nx,ny+1);
q=zeros(2,nx,ny+1);
qn=zeros(2,nx,ny+1);
qpvo=zeros(2,nx,ny+1);

for j=1:ny+1
    Q(1,:,j) = real(ifft(squeeze(qpvn(1,:,j))));
    Q(2,:,j) = real(ifft(squeeze(qpvn(2,:,j))));
end
for i=1:nx
    q(1,i,:) = CosineT(squeeze(Q(1,i,:))',ny);
    q(2,i,:) = CosineT(squeeze(Q(2,i,:))',ny);
end
for j=1:ny+1
    qn(1,:,j)=fft(squeeze(q(1,:,j)));
    qn(2,:,j)=fft(squeeze(q(2,:,j)));
    qn(:,nx/2+1,j)=0.0;
    for i=1:nx
        qpvn(:,i,j) = qn(:,i,j)*exp(-(nux*(kx(i))^p+nuy*(j-1)^p)*dt);
    end
end

q(:,:,:)=0.0;
for j=1:ny+1
    q(1,:,j) = real(ifft(squeeze(qpvn(1,:,j))));
    q(2,:,j) = real(ifft(squeeze(qpvn(2,:,j))));
end
Q(:,:,:)=0.0;
for i=1:nx
    Q(1,i,:) = iCosineT(squeeze(q(1,i,:))',ny);
    Q(2,i,:) = iCosineT(squeeze(q(2,i,:))',ny);
end
for j=1:ny+1
    qpvo(1,:,j)=fft(squeeze(Q(1,:,j)));
    qpvo(2,:,j)=fft(squeeze(Q(2,:,j)));
end

qpvo(:,nx/2+1,:)=0.0;