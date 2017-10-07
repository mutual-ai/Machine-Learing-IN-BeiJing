%����Ϊ��3��Ľ�ͨ����������Ϊһ�������Ϊ50��
W=W1+W2+W3;
[C,W_center]=kmeans(W,50);

%����50��֮��ľ���
for i=1:50
    for j=1:50
        x1=W_center(i,1);
        x2=W_center(j,1);
        y1=W_center(i,2);
        y2=W_center(j,2);
        z1=W_center(i,3);
        z2=W_center(j,3);
        distance(i,j)=sqrt((x1-x2).^2+(y1-y2).^2+(z1-z2).^2);
    end
end
%��50��ľ����һ��
dist=mapminmax(distance,0,1);
%ѡȡ��ֵ����ʹ��dist������0-1�ڽӾ���
for i=1:50
    for j=1:50
        if dist(i,j)<=0.1      %�Լ�ѡ����ֵ
            dist1(i,j)=1;
        else dist1(i,j)=0;
        end
    end
end

%����·�ξ��룬��ÿһ��·��Χ��15��·����(����·���������)�Ž������У�
for i=1:127049
    road_name(i,1)=flowdata1(i,1);
    road_name(i,2:16)=i+1:i+15;
end
for i=127034:127049
    road_name(i,1)=flowdata1(i,1);
    road_name(i,2:16)=127035:127049;
end


%���ݾ���ģʽ�����µ�ɸѡ,�ֲ�ʽ���㣬�ڴ�̫С
for i=1:127049
    class1=C(i,1);
    for j=i+1:i+15
        class2=C(j,1);
        if dist1(class1,class2)==0
        road_name(i,j-i+1)=0;
        end
    end
end


%��������ܶȷֲ�����
means=20;
x=0:300;
y=poisspdf(x,means);
sum(y(xt:301));%���㵱�����������岴�ɷֲ��еĸ��ʿ���ֵ

%�����ڴ�ԭ�򣬽�һ��96��ʱ��α��4��ʱ���
for i=1:127049
        t1(i)=sum(V1(i,1:24));
        t2(i)=sum(V1(i,25:48));
        t3(i)=sum(V1(i,49:72));
        t4(i)=sum(V1(i,73:96));
        T1(i,1)=t1(i);
        T1(i,2)=t2(i);
        T1(i,3)=t3(i);
        T1(i,4)=t4(i);
end

for i=1:127049
        t1(i)=sum(V2(i,1:24));
        t2(i)=sum(V2(i,25:48));
        t3(i)=sum(V2(i,49:72));
        t4(i)=sum(V2(i,73:96));
        T2(i,1)=t1(i);
        T2(i,2)=t2(i);
        T2(i,3)=t3(i);
        T2(i,4)=t4(i);
end

for i=1:127049
        t1(i)=sum(V3(i,1:24));
        t2(i)=sum(V3(i,25:48));
        t3(i)=sum(V3(i,49:72));
        t4(i)=sum(V3(i,73:96));
        T3(i,1)=t1(i);        
        T3(i,2)=t2(i);
        T3(i,3)=t3(i);
        T3(i,4)=t4(i);
end

T=(1/3)*(T1+T2+T3);
max(max(T));%���ڼ���������ͨ��̫�Ӵ����ǳ���100����Сx�ķֲ�
T=0.1*T;


%������·�Σ�ͬһʱ����Ƚϣ������ܶȷֲ�
x=1:308;
V4=flowdata4(:,4:99);
for i=1:127049
        t1(i)=sum(V4(i,1:24));
        t2(i)=sum(V4(i,25:48));
        t3(i)=sum(V4(i,49:72));
        t4(i)=sum(V4(i,73:96));
        T4(i,1)=t1(i);        
        T4(i,2)=t2(i);
        T4(i,3)=t3(i);
        T4(i,4)=t4(i);
end
T4=0.1*T4;
T4(i,j);%��i����·��jʱ�̵�ʮ��֮һ������

%�����·���ھӵ�·�Ĳ��ɷֲ��ľ�ֵ��Ϊ�˼�С����������ֵ����10������i����·����j��ʱ�̣�
for i=1:127049
    for j=1:4
        %V4(i,j);
        count=0;
        sum1(i,j)=0;
        for k=2:16%�����·���ھӵ�·�ĳ������ͣ���i����·����j��ʱ�̣�
            if road_name(i,k)~=0
                road_num=road_name(i,k);
                sum1(i,j)=sum1(i,j)+T(road_num,j);                
                count=count+1;
            end    
        end
        roadr_means(i,j)=sum(i,j)/count;
        if  roadr_means(i,j)==0
            roadr_means(i,j)=0.1;
        end    
    end
end    
roadr_means(isnan(roadr_means)) = 0.1;

%�����·����ʷ���ݵĲ��ɷֲ��ľ�ֵ��Ϊ�˼�С����������ֵ����10������i����·����j��ʱ�̣�
roadt_means=T;

%���鼯
%
%
%����poisson�ֲ����ɴ�֣�����Ϊ������ʮ���Ӵ����ǿ���Ϊ������̬�ֲ���
t0=0.1*T3;%��i����·��jʱ�̵�ʮ��֮һ������
t0=round(t0);
%����ϵ��Ϊ0.5
z=0.5;
for i=1:127049
    for j=1:4
        if t0(i,j)==0 t0(i,j)=1; end
        Pr=poisspdf(x,roadr_means(i,j));%�ھӵ�·
        Pt=poisspdf(x,roadt_means(i,j));%��ʷ����
        point0(i,j)=z*(sum(Pr(t0(i,j):308)))+(1-z)*(sum(Pt(t0(i,j):308)));
    end        
end     

%�ж��쳣��·,ͳ��һ�±��쳣��·�θ���������·��ֵ
count0_1=0;
count0_2=0;
for i=1:127049
    for j=1:4
        if point0(i,j)<=0.001
            judge0(i,j)=1;
            count0_1=count0_1+1;
        else if point0(i,j)<0.05
            judge0(i,j)=2;
            count0_2=count0_2+1;
            else judge0(i,j)=0;
            end
        end
    end
end


%Ԥ�⼯
%
%
%����poisson�ֲ����ɴ�֣�����Ϊ������ʮ���Ӵ����ǿ���Ϊ������̬�ֲ���
t=T4;%��i����·��jʱ�̵�ʮ��֮һ������
t=round(t);
%����ϵ��Ϊ0.5
z=0.5;
for i=1:127049
    for j=1:4
        if t(i,j)==0 t(i,j)=1; end
        Pr=poisspdf(x,roadr_means(i,j));%�ھӵ�·
        Pt=poisspdf(x,roadt_means(i,j));%��ʷ����
        point(i,j)=z*(sum(Pr(t(i,j):308)))+(1-z)*(sum(Pt(t(i,j):308)));
    end        
end     

%�ж��쳣��·,ͳ��һ�±��쳣��·�θ���������·��ֵ
count1=0;
count2=0;
for i=1:127049
    for j=1:4
        if point(i,j)<=0.001
            judge(i,j)=1;
            count1=count1+1;
            disp(i);
            disp(j);
        else if point(i,j)<0.05
            judge(i,j)=2;
            count2=count2+1;
            else judge(i,j)=0;
            end
        end
    end
end
