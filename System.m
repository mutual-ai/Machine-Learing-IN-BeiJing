%定义为将3天的交通流合起来成为一个矩阵分为50类
W=W1+W2+W3;
[C,W_center]=kmeans(W,50);

%计算50类之间的距离
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
%将50类的距离归一化
dist=mapminmax(distance,0,1);
%选取阈值，并使得dist矩阵变成0-1邻接矩阵
for i=1:50
    for j=1:50
        if dist(i,j)<=0.1      %自己选定阈值
            dist1(i,j)=1;
        else dist1(i,j)=0;
        end
    end
end

%按照路段距离，将每一条路周围的15条路段名(根据路段起点名字)放进矩阵中，
for i=1:127049
    road_name(i,1)=flowdata1(i,1);
    road_name(i,2:16)=i+1:i+15;
end
for i=127034:127049
    road_name(i,1)=flowdata1(i,1);
    road_name(i,2:16)=127035:127049;
end


%根据聚类模式进行新的筛选,分布式计算，内存太小
for i=1:127049
    class1=C(i,1);
    for j=i+1:i+15
        class2=C(j,1);
        if dist1(class1,class2)==0
        road_name(i,j-i+1)=0;
        end
    end
end


%计算概率密度分布函数
means=20;
x=0:300;
y=poisspdf(x,means);
sum(y(xt:301));%计算当今流量在总体泊松分布中的概率可能值

%由于内存原因，将一天96个时间段变成4个时间段
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
max(max(T));%由于计算出的最大交通量太庞大，我们除以100来缩小x的分布
T=0.1*T;


%跟其他路段，同一时刻相比较，概率密度分布
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
T4(i,j);%第i条道路，j时刻的十分之一车流量

%计算该路段邻居道路的泊松分布的均值（为了减小计算量，均值除以10）（第i条道路，第j个时刻）
for i=1:127049
    for j=1:4
        %V4(i,j);
        count=0;
        sum1(i,j)=0;
        for k=2:16%求出该路段邻居道路的车流量和（第i条道路，第j个时刻）
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

%计算该路段历史数据的泊松分布的均值（为了减小计算量，均值除以10）（第i条道路，第j个时刻）
roadt_means=T;

%检验集
%
%
%根据poisson分布生成打分，（因为数据量十分庞大，我们可认为近似正态分布）
t0=0.1*T3;%第i条道路，j时刻的十分之一车流量
t0=round(t0);
%设置系数为0.5
z=0.5;
for i=1:127049
    for j=1:4
        if t0(i,j)==0 t0(i,j)=1; end
        Pr=poisspdf(x,roadr_means(i,j));%邻居道路
        Pt=poisspdf(x,roadt_means(i,j));%历史数据
        point0(i,j)=z*(sum(Pr(t0(i,j):308)))+(1-z)*(sum(Pt(t0(i,j):308)));
    end        
end     

%判断异常道路,统计一下报异常的路段个数并返回路段值
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


%预测集
%
%
%根据poisson分布生成打分，（因为数据量十分庞大，我们可认为近似正态分布）
t=T4;%第i条道路，j时刻的十分之一车流量
t=round(t);
%设置系数为0.5
z=0.5;
for i=1:127049
    for j=1:4
        if t(i,j)==0 t(i,j)=1; end
        Pr=poisspdf(x,roadr_means(i,j));%邻居道路
        Pt=poisspdf(x,roadt_means(i,j));%历史数据
        point(i,j)=z*(sum(Pr(t(i,j):308)))+(1-z)*(sum(Pt(t(i,j):308)));
    end        
end     

%判断异常道路,统计一下报异常的路段个数并返回路段值
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
