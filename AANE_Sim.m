%% SDM 2017 Accelerated Attributed Network Embedding
%dl = 100; lambda = 2.8; rho = 80; Net = CombG;
function H = AANE_Sim(Net,SA,dl,lambda,rho)
N = size(Net,1);
Net(1:N+1:N^2) = 0;
% tol = 10^-3*N;
maxIter = 2;
[H,~] = svds(Net(:,1:8*dl),dl); %
H = H'; % Transpose for speedup
Z = H;
Net = lambda*Net;
%% First update H
XTX = Z*Z'*2; % Transposed
sumS = Z*SA'*2;
for i = 1:N
    Neighbor = Z(:,Net(:,i)~=0); % z_j^k
    normhi_zj = sum((bsxfun(@minus,Neighbor,H(:,i))).^2).^.5; % norm of h_i^k-z_j^k
    Wij = Net(i,Net(:,i)~=0);
    NNZero = normhi_zj>0; %10^-12 % Number of non-zero
    normhi_zj = Wij(NNZero)./normhi_zj(NNZero);
    if sum(NNZero) % ~isempty(normhi_zj)
        H(:,i)=((sumS(:,i)+sum(bsxfun(@times,Neighbor(:,NNZero),normhi_zj),2)+rho*(Z(:,i)))'/(XTX+(sum(normhi_zj)+rho)*eye(dl)))';
    else 
        H(:,i) = ((sumS(:,i)+rho*(Z(:,i)))'/(XTX+rho*eye(dl)))';
    end
end

%% Interation starts
U = zeros(dl,N);
for iter = 1:maxIter-1
    %% Update Z
    XTX = H*H'*2;
    sumS = H*SA'*2;
    for i = 1:N
        Neighbor = H(:,Net(:,i)~=0);
        normzi_hj = sum((bsxfun(@minus,Neighbor,Z(:,i))).^2).^.5;
        Wij = Net(i,Net(:,i)~=0);
        NNZero = normzi_hj>0;
        normzi_hj = Wij(NNZero)./normzi_hj(NNZero);
        if sum(NNZero)
            Z(:,i)=((sumS(:,i)+sum(bsxfun(@times,Neighbor(:,NNZero),normzi_hj),2)+rho*(H(:,i)+U(:,i)))'/(XTX+(sum(normzi_hj)+rho)*eye(dl)))';
        else
            Z(:,i) = ((sumS(:,i)+rho*(H(:,i)+U(:,i)))'/(XTX+rho*eye(dl)))';
        end
    end
    
    U=U+H-Z; % Updata U % U(:,LocalIndex)=U(:,LocalIndex)+H(:,LocalIndex)-Z(:,LocalIndex); %Locally Update U
    %% Update H
    XTX = Z*Z'*2; % Transposed
    sumS = Z*SA'*2;
    for i = 1:N
        Neighbor = Z(:,Net(:,i)~=0);
        normhi_zj = sum((bsxfun(@minus,Neighbor,H(:,i))).^2).^.5; % norm of h_i^k-z_j^k
        Wij = Net(i,Net(:,i)~=0);
        NNZero = normhi_zj>0; %10^-12
        normhi_zj = Wij(NNZero)./normhi_zj(NNZero);
        if sum(NNZero) % ~isempty(normhi_zj)
            H(:,i)=((sumS(:,i)+sum(bsxfun(@times,Neighbor(:,NNZero),normhi_zj),2)+rho*(Z(:,i)-U(:,i)))'/(XTX+(sum(normhi_zj)+rho)*eye(dl)))';
        else
            H(:,i) = ((sumS(:,i)+rho*(Z(:,i)-U(:,i)))'/(XTX+rho*eye(dl)))';
        end
    end
% H=H';
% [~,F1microtmp]=Performance(H(1:n1,:),H(n1+1:n1+n2,:),Label(Group1,:),Label(Group2,:)) %
% H=H';
end
H=H';