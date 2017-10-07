%非负矩阵分解，直接调用nnmf函数
V1=flowdata1(:,4:99);
V2=flowdata2(:,4:99);
V3=flowdata3(:,4:99);

r=3;%设置模式数量
[W1,H1]=nnmf(V1,r);
[W2,H2]=nnmf(V2,r);
[W3,H3]=nnmf(V3,r);

