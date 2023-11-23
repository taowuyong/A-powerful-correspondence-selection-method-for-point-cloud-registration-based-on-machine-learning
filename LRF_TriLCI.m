function [V] = LRF_TriLCI(KNN,RR,d,keypoint)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[h l]=size(KNN);
if h<4
    V=eye(3);
else
    KNNc=KNN(1:4,:);
    [U S V]=svd((KNNc-ones(4,1)*mean(KNNc))'*(KNNc-ones(4,1)*mean(KNNc)));
    normal=V(:,3);
    for j=1:1000
        dprojection=abs((KNN-ones(h,1)*keypoint)*normal);
        iidx=find(dprojection<0.13*RR);                %parameter can be adjuseed
        KNNcc=KNN(iidx,:);
        [h2 l2]=size(KNNcc);
        [U1 S1 V1]=svd((KNNcc-ones(h2,1)*mean(KNNcc))'*(KNNcc-ones(h2,1)*mean(KNNcc)));
        if norm(abs(V1(:,3))-abs(normal))<0.00001
            break
        end
        normal=V1(:,3);
    end
    qp=ones(h,1)*keypoint-KNN;
    qp=sum(qp);
    if qp*normal>=0
        V3=normal;
    else
        V3=-normal;
    end
    MPvector=[];
    for j=1:h
        Pvector=(KNN(j,:)-keypoint)'-((KNN(j,:)-keypoint)*V3)*V3;
        MPvector=[MPvector Pvector];
    end
    clear w1 w2 w;
    for j=1:h
        w1(j)=(RR-d(j))^2;
        w2(j)=((KNN(j,:)-keypoint)*V3)^2;
        w(j)=w1(j)*w2(j);
    end
    V1=zeros(3,1);
    for j=1:h
        V1=V1+w(j)*MPvector(:,j);
    end
    V1=V1/norm(V1);
    V2=cross(V1,V3);
    V=[V1 V2 V3]; 
end
end
