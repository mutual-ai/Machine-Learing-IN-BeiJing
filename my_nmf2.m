%�Ǹ�����ͼ��ֽ�
X=imread('001.jpg');
V=X;
imshow(mat2gray(V));

[i u]=size(V);                          %����V�Ĺ��
r=100;                                  %���÷ֽ�������
W=rand(i,r);                            %��ʼ��WH��Ϊ�Ǹ���
H=rand(r,u);
maviter=100;                            %����������
T=max(max(abs(V-W*H)));
while true
    W=W.*((V./(W*H))*H');               %ע�������������ʽ�����е��Ƕ�Ӧ��
    W=W./(ones(i,1)*sum(W));   
    H=H.*(W'*(V./(W*H)));
    if max(max(abs(V-W*H)))
        T=max(max(abs(V-W*H)));
        if T<35
            break;
        end
    end
end

img_V=W*H;
figure;
imshow(mat2gray(img_V));