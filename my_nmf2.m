%非负矩阵图像分解
X=imread('001.jpg');
V=X;
imshow(mat2gray(V));

[i u]=size(V);                          %计算V的规格
r=100;                                  %设置分解矩阵的秩
W=rand(i,r);                            %初始化WH，为非负数
H=rand(r,u);
maviter=100;                            %最大迭代次数
T=max(max(abs(V-W*H)));
while true
    W=W.*((V./(W*H))*H');               %注意这里的三个公式和文中的是对应的
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