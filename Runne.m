d = 100;
OracPro = 0.88; %  Procentage of knowledge that oracle knows
beta1 = 0.67; % For network and attributes
gamma = 0.22;
B = 9; % Number of topic nodes
lambda = 5;
alpha = 1+sqrt(log(2/0.001)/2);
load('BlogCatalog')
[N,m] = size(Attributes);

Indices = randi(25,N,1); % Indices for cross-validation
Group2 = find(Indices >= 21); % testing set
Group1 = find(Indices <= 20); % Group1 = ~Group2;
n1 = length(Group1);
n2 = length(Group2);
n = n1+n2;
InitG = sparse(Network([Group1;Group2],[Group1;Group2]));
InitA = sparse(Attributes([Group1;Group2],:));
%% Oracle Info
[HideG, ~] = datasample(find(triu(InitG,1)~=0),round(0.5*(OracPro*nnz(InitG)-n)),'Replace',false);
[HideA, ~] = datasample(find(InitA~=0),round(OracPro*nnz(InitA)),'Replace',false);
OracleS = CosSim(InitA);% Original SA
CombA = InitA; % Initial A
CombA(HideA) = 0;
SA = CosSim(CombA);
DXInv = spdiags(full(sum(SA,2)).^(-.5),0,n,n);
LA = DXInv*SA*DXInv;
LA = .5*(LA+LA');
CombG = InitG; % Initial G
CombG(HideG) = 0;
CombG = triu(CombG,1) + triu(CombG)';
K = ceil(0.1*(nnz(CombG)-n));%
%% Spectural embedding
opts.v0 = rand(n,1);
normalG = bsxfun(@rdivide,CombG,sum(CombG.^2).^.5); % Normalize
SG = normalG'*normalG; % +10^-6*eye(n)
SG(1:n+1:n^2) = 1;
DXInv = spdiags(full(sum(SG+10^-6*speye(n),2)).^(-.5),0,n,n);
LG = DXInv*(SG+10^-6*speye(n))*DXInv;
[UG,~] = eigs(.5*(LG+LG'),d,'LM',opts);
[F1macro, F1micro] = Performance(UG(1:n1,:),UG(n1+1:n1+n2,:),Label(Group1),Label(Group2));
disp('Initial spectural embedding on pure network:')
F1micro
[UA,~] = eigs(.5*(LA+LA'),d,'LM');
[iniH,~] = eigs(norLap(beta1*SG+(1-beta1)*SA),d,'LM',opts); % 
[F1macro, F1micro] = Performance(iniH(1:n1,:),iniH(n1+1:n1+n2,:),Label(Group1,:),Label(Group2,:));
disp('Initial NEEC embedding on attributed network:')
F1micro
SH = CosSim(iniH);
SH(1:n+1:n^2) = -inf;
DistGA = abs(CosSim(UG)-CosSim(UA));

%% AANE
H = AANE_Sim(SG,SA,180,.55,8);
[F1macro, F1micro] = Performance(H(1:n1,1:d),H(n1+1:n1+n2,1:d),Label(Group1),Label(Group2));
disp('Initial AANE embedding on attributed network:')
F1micro

disp('Learning expert cognition with number of queries K = 4185, running...')
[~,~,~,kmeD] = kmedoids(UA,B,'Distance' ,'cosine'); % ,'Algorithm','small'
[~,SetCen] = min(kmeD); % Set of nodes in the center
kmeD = 1 - SH(:,SetCen);
minDist = min(kmeD,[],2);
SetMar = setdiff(1:n,SetCen);
Xak = [iniH(SetMar,1:d),minDist(SetMar),mean(kmeD(SetMar,:),2)]; % ,sum(kmeD(SetMar,:),2)
armIdx = kmeans(iniH,d);
CTCI = zeros(d+2,d+2,d);
CTc = zeros(d+2,d); % Current C^T * c
for armi = 1:d
    CTCI(:,:,armi) = lambda*eye(d+2);  % Current (C^T * C + I)
end
ExpectRawrd = alpha*(dot(Xak',Xak')').^.5; % theta=0
for k = 1:K
    if k == 1
        maxIdx = randi(n-B);
    else
        [~,maxIdx] = max(ExpectRawrd);
    end
    i = SetMar(maxIdx);
    Ck_1 = Xak(maxIdx,:); % matrix that collects the xa,k that already selected
    [~,LabIdx] = max(OracleS(i,SetCen));
    j = SetCen(LabIdx);
    CombG(i,j) = gamma+CombG(i,j);
    CombG(j,i) = CombG(i,j);
    cReward = DistGA(i,j);
    SetMar(maxIdx) = [];
    Xak(maxIdx,:) = [];
    ExpectRawrd(maxIdx) = [];
    CTCI(:,:,armIdx(i)) = CTCI(:,:,armIdx(i)) + Ck_1'*Ck_1; % curreqnt (C^T * C + I)
    CTc(:,armIdx(i)) = CTc(:,armIdx(i)) + cReward*Ck_1';
    UpdaArm = armIdx(SetMar)==armIdx(i);
    InvCTCI = pinv(CTCI(:,:,armIdx(i)));
    ExpectRawrd(UpdaArm) = Xak(UpdaArm,:)*(InvCTCI*CTc(:,armIdx(i)))+alpha*(dot(Xak(UpdaArm,:)',InvCTCI*Xak(UpdaArm,:)')').^.5; % theta
end
normalG = bsxfun(@rdivide,CombG,sum(CombG.^2).^.5); % Normalize
SG = normalG'*normalG; % +10^-6*eye(n)
SG(1:n+1:n^2) = 1;
DXInv = spdiags(full(sum(SG+10^-6*speye(n),2)).^(-.5),0,n,n);
LG = DXInv*(SG+10^-6*speye(n))*DXInv;
[UG,~] = eigs(.5*(LG+LG'),d,'LM',opts); %

[F1macro, F1micro] = Performance(UG(1:n1,:),UG(n1+1:n1+n2,:),Label(Group1),Label(Group2));
disp('With expert cognition, spectural embedding on pure network:')
F1micro

[H,~] = eigs(norLap(beta1*SG+(1-beta1)*SA),d,'LM',opts); %
[F1macro, F1micro] = Performance(H(1:n1,:),H(n1+1:n1+n2,:),Label(Group1,:),Label(Group2,:));
disp('With expert cognition, NEEC embedding on attributed network:')
F1micro

%% AANE
H = AANE_Sim(SG,SA,180,.55,8);
[F1macro, F1micro] = Performance(H(1:n1,1:d),H(n1+1:n1+n2,1:d),Label(Group1),Label(Group2));
disp('With expert cognition, AANE embedding on attributed network:')
F1micro
