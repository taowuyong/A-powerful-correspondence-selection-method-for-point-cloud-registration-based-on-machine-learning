function [DV] = Tri_LCI(KNN,keypoint,V,RR)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[h l]=size(KNN);
KNNt=(KNN-ones(h,1)*keypoint)*V;
wbin=15;              %%  bin number
Lbin=2*RR/wbin;
%projection on xy plane
for s=1:wbin
    for t=1:wbin
        idd=find(KNNt(:,1)>=-RR+(s-1)*Lbin & KNNt(:,1)<-RR+s*Lbin & KNNt(:,2)>=-RR+(t-1)*Lbin & KNNt(:,2)<-RR+t*Lbin);
        if isempty(idd)
            Dxy(s,t)=0;
        else
            Dxy(s,t)=max(KNNt(idd,3))/(RR);
        end
    end
end
%projection on yz plane
idx1=find(KNNt(:,1)<0);
idx2=find(KNNt(:,1)>=0);
KNNt1=KNNt(idx1,:);
KNNt2=KNNt(idx2,:);
KNNt1=[abs(KNNt1(:,1)) KNNt1(:,2) abs(KNNt1(:,3))];
KNNtyz=[KNNt1;KNNt2];
for s=1:wbin
    for t=1:wbin
        idd=find(KNNtyz(:,2)>=-RR+(s-1)*Lbin & KNNtyz(:,2)<-RR+s*Lbin & KNNtyz(:,3)>=-RR+(t-1)*Lbin & KNNtyz(:,3)<-RR+t*Lbin);
        if isempty(idd)
            Dyz(s,t)=0;
        else
            Dyz(s,t)=max(KNNtyz(idd,1))/(RR);
        end
    end
end
%projection on xz plane
idx1=find(KNNt(:,2)<0);
idx2=find(KNNt(:,2)>=0);
KNNt1=KNNt(idx1,:);
KNNt2=KNNt(idx2,:);
KNNt1=[KNNt1(:,1) abs(KNNt1(:,2)) abs(KNNt1(:,3))];
KNNtxz=[KNNt1;KNNt2];
for s=1:wbin
    for t=1:wbin
        idd=find(KNNtxz(:,1)>=-RR+(s-1)*Lbin & KNNtxz(:,1)<-RR+s*Lbin & KNNtxz(:,3)>=-RR+(t-1)*Lbin & KNNtxz(:,3)<-RR+t*Lbin);
        if isempty(idd)
            Dxz(s,t)=0;
        else
            Dxz(s,t)=max(KNNtxz(idd,2))/(RR);
        end
    end
end
DV=[Dxy(:)' Dyz(:)' Dxz(:)'];
end










