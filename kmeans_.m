%����Ϊ��3��Ľ�ͨ����Ϊ3��
C(:,1)=kmeans(W1,3);
C(:,2)=kmeans(W2,3);
C(:,3)=kmeans(W3,3);

%����W1�����Ժ�Ľ��
for i=1:20000
    if C(i,1)==1
        plot3(W1(i,1),W1(i,2),W1(i,3),'b+');
        hold on;
    elseif C(i,1)==2
        plot3(W1(i,1),W1(i,2),W1(i,3),'r+');
        hold on;
    else
        plot3(W1(i,1),W1(i,2),W1(i,3),'g+');
        hold on;
    end
end
    