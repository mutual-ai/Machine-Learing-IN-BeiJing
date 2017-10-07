V=V1;
[n,m]=size(V);
k=min(n,m);%r的数值不能大于矩阵中的最小的行或列
for count = 2:k
    r=count;
    W=rand(n,r);
    H=rand(r,m);
    for loop=1:10%对r目前的数值进行10次迭代
        L1=W*H*H';
        L2=W'*W*H;
        if (all(L1(:))==1 && any(L2(:))==1)
        W=W.*((V*H')./(W*H*H')); %迭代更新W          
        %W=W./(ones(m,1)*sum(W));    
        H=H.*((W'*V)./(W'*W*H));%迭代更新H
        end
    end
    final_V=W*H; 
    %判断r为多少的时候是合适的(空间夹角余弦值)
    cal(k)=sum(sum((final_V.*V)/(norm(final_V)*norm(V))));
end    
 




