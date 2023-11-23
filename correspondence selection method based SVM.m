%%correspondence selection method based SVM
pcloud1=pcread('E:\compile document\matlab\data\PPC calculation\livingroom10.ply');
PC1=pcloud1.Location;
pcloud2=pcread('E:\compile document\matlab\data\PPC calculation\livingroom11.ply');
PC2=pcloud2.Location;
[n m]=size(PC2);
% plot3(PC1(:,1),PC1(:,2),PC1(:,3),'.b','MarkerSize',1);
% hold on;
% plot3(PC2(:,1),PC2(:,2),PC2(:,3),'.r','MarkerSize',1);
% set(gca,'DataAspectRatio',[1 1 1]);
% axis off
pr=0.006956074;                     %ICL-NUIM dataset
keypointcloud1 = pcdownsample(pcloud1,'gridAverage',7*pr);     %%Kepoint extraction by Uniform sampling
keypoint1=keypointcloud1.Location;
[n1 m1]=size(keypoint1);
RR=15*pr;
[idx1,dist1]=rangesearch(PC1,keypoint1,RR);
MV1=[];
MDV1=[];
for i=1:n1
    KNN=PC1(idx1{i},:);
    d=dist1{i};
    [V] = LRF_TriLCI(KNN,RR,d,keypoint1(i,:));
    MV1=[MV1;V];      %%save LRF
    [DV] = Tri_LCI(KNN,keypoint1(i,:),V,RR);
    MDV1=[MDV1;DV];    %%save TriLCI descriptor
end
keypointcloud2 = pcdownsample(pcloud2,'gridAverage',7*pr);       %%Kepoint extraction by Uniform sampling
keypoint2=keypointcloud2.Location;
[n2 m2]=size(keypoint2);
[idx2,dist2]=rangesearch(PC2,keypoint2,RR);
MV2=[];
MDV2=[];
for i=1:n2
    KNN=PC2(idx2{i},:);
    d=dist2{i};
    [V] = LRF_TriLCI(KNN,RR,d,keypoint2(i,:));
    MV2=[MV2;V];    %%save LRF
    [DV] = Tri_LCI(KNN,keypoint2(i,:),V,RR);
    MDV2=[MDV2;DV];    %%save TriLCI descriptor
end
[idxx distt]=knnsearch(MDV1,MDV2,'k',2);
Mmatch=[];
for i=1:n2
    if distt(i,1)/distt(i,2)<=0.9
        match=[idxx(i,1) i];
        Mmatch=[Mmatch;match];
    end
end
[n3 m3]=size(Mmatch);   %% Mmatch is the initial correspondence set. 
MFc=[];
for i=1:n3
    P1=keypoint1(Mmatch(i,1),:);
    Q1=keypoint2(Mmatch(i,2),:);
    F1=[];
    F2=[];
    for j=1:n3
        if j==i
            continue;
        else
            P2=keypoint1(Mmatch(j,1),:);
            Q2=keypoint2(Mmatch(j,2),:);
            d1=abs(norm(P1-P2)-norm(Q1-Q2));
            V11=MV1(3*Mmatch(i,1)-2:3*Mmatch(i,1),:);
            V12=MV2(3*Mmatch(i,2)-2:3*Mmatch(i,2),:);
            V21=MV1(3*Mmatch(j,1)-2:3*Mmatch(j,1),:);
            V22=MV2(3*Mmatch(j,2)-2:3*Mmatch(j,2),:);
            R1=V12*V11';
            R2=V22*V21';
            d2=abs(norm(R1*P1'-Q1')-norm(R2*P2'-Q2'));
            CC1=exp(-d1*d1/(2*1*pr*1*pr));            
            CC2=exp(-d2*d2/(2*40*pr*40*pr));            %parameter
        end
        F1=[F1 CC1];
        F2=[F2 CC2];
    end
    F=[];
    for i=1:n3-1
        F=[F min([F1(i) F2(i)])];     %%minpooling
    end
    F=sort(F,'descend');
    dim=40;                        %dimensionality of the feature vector
    Fc=F(:,1:dim);                 %%final feature vector
    MFc=[MFc;Fc];                  %%feature vectors of all correspondences
end
tsvm = load('svm.mat');               %%SVM classifier
classification=tsvm.svmModel.ClassNames;
CVSVMModel = crossval(tsvm.svmModel);
test=MFc;
result = tsvm.svmModel.predict(test);
idre=find(result==1);
Cinlier=Mmatch(idre,:);            %%the selected correspondences
%%using the selected correspondence to calculate the transformation
[n5 m5]=size(Cinlier);
A=keypoint2(Cinlier(:,2),:);
Y=keypoint1(Cinlier(:,1),:);
uA=[mean(A(:,1)) mean(A(:,2)) mean(A(:,3))];
uY=[mean(Y(:,1)) mean(Y(:,2)) mean(Y(:,3))];
H=zeros(3);
for j=1:n5
    H=H+(A(j,:)-uA)'*(Y(j,:)-uY);
end
[U S V]=svd(H);
D=diag([1 1 det(U*V')]);
R=V*D*U';
t=uY-uA*R';
T=[R t';zeros(1,3) 1];     %%transformation
PC2t=PC2*R'+ones(n,1)*t;
% plot3(PC1(:,1),PC1(:,2),PC1(:,3),'.b','MarkerSize',1);
% hold on;
% plot3(PC2t(:,1),PC2t(:,2),PC2t(:,3),'.r','MarkerSize',1);
% hold on;
% set(gca,'DataAspectRatio',[1 1 1]);





