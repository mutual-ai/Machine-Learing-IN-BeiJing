V=V1;
[n,m]=size(V);
k=min(n,m);%r����ֵ���ܴ��ھ����е���С���л���
for count = 2:k
    r=count;
    W=rand(n,r);
    H=rand(r,m);
    for loop=1:10%��rĿǰ����ֵ����10�ε���
        L1=W*H*H';
        L2=W'*W*H;
        if (all(L1(:))==1 && any(L2(:))==1)
        W=W.*((V*H')./(W*H*H')); %��������W          
        %W=W./(ones(m,1)*sum(W));    
        H=H.*((W'*V)./(W'*W*H));%��������H
        end
    end
    final_V=W*H; 
    %�ж�rΪ���ٵ�ʱ���Ǻ��ʵ�(�ռ�н�����ֵ)
    cal(k)=sum(sum((final_V.*V)/(norm(final_V)*norm(V))));
end    
 




