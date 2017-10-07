机器学习
====

这次来北京，主要还是想要自己能学一些东西，所以还是使用Matlab代码，所以我这里暂时就先用Matlab算法来计算，如果不行，我们再使用R语言来去计算

* 数据挖据的功能及应用  

    * 分类： 决策树、支持向量机、K-近邻、贝叶斯
    * 聚类分析： K-means；K-中心；DBSan
    * 预测： 线性回归；非线性回归
    * 关联分析： 统计

* 数据挖掘的流程  
    * 商业理解（需求理解）
    * 数据理解
    * **数据准备**（构造数据最终集合、净化和转换数据）
    * **模型建立**（选择和应用各种模型技术，矫正和优化个证模型参数）
    * 模型评估
    * 模型发布  

* 1.数据清洗  
    * 清洗方法：解决不完整数据、错误噪声（偏差分析、回归分析、规则库）
    * 错误噪声的解决方法： ***分箱！！！*** 划分成等深的箱，用箱平均值平滑或者用边界平滑
    * 重复记录的数据：***套牌车，这可以是一个很好的课题*** 实时

    * 实际应用：车辆轨迹数据清洗，北京市12712辆出租车，2011年11月11日之后的30天的GPS采样数据，共有333651069个，平均采样点数量26349

    * 不真实点（超出北京的经纬范围），重复时间点（同一时间的多个点），高速点（大于90km/h），距离过长点（采样时间内超过2km），时间过长点（处于行驶状态的超过10分钟点未动），停住点（30min内没有50m的运动），等乘客点

~~~
下面是一段Matlab代码：
data=allgroup;
j=1;
for i=1:137085
    if (data(i,7)>=426516480&&data(i,7)<=432783360&&data(i,8)>=145244160&&data(i,8)<=153354240)==0
        problemposition(j)=i;
        j=j+1;
    end
end

j=1;
for i=1:137085
    if data(i,9)>=90
        problemvelocity(j)=i;
        
        j=j+1;
    end
end


%距离超长点，大于2km
j=0;
for i=1:137084
    x1=data(i,5)*111;
    x2=data(i+1,5)*111;
    y1=data(i,6)*111;
    y2=data(i+1,6)*111;
    distance=sqrt((x1-x2).^2+(y1-y2).^2);
    if (distance>=2 && data(i,3)==data(i+1,3))
        problemdis(j)=i;
        j=j+1;
    end   
end    
~~~


 * 匹配问题产生：将GPS的点要映射到地图上去  
        * Step 1：针对每个采样点找出一系列采样路段
        * Step 2：针对每一个相邻的点都要处理一下，从而找出一条全局最可能的路径
        * Step 3：在时空分析匹配算法，将原始的GPS轨迹点匹配到数字地图上的路径

    * 道路匹配算法（ST-Matching）  
        * 先计算出所有候选点
        * 然后，根据点的200m半径，计算出其所有的候选边
        * 然后根据空间因素权重和时间因素权重Fs*Ft，我们就可以计算出一条最佳匹配路径


  
~~~
我在github上fork了几个比较牛逼的java代码，别人已经实现了的
~~~

* 2.主成分分析和非负矩阵分解  
    * 两者关系  
        * 矩阵分解（A=BC）来解决实际问题：PCA（主成分分析），ICA（独立成分分析），SVD（奇异值分析），VQ（矢量量化）
        * VQ A=BC
        * PCA 将几个人完整的脸（特征脸）加减压成一张脸，可以将基图像的量降下来
        * NMF 加了非负的约束（将PCA中负的去掉），会发现某一部分正在描述人的某一部分的结构，比如人的眼睛，鼻子嘴巴

    * 主成分分析 （Pearson 1901） 
        * 简化数据集的技术
        * 降维技术：线性变换x->y上面，从而去掉x
        * 数据标准化，计算相关矩阵，计算特征值和特征向量，求主成分
            * 特征值对应椭圆轴的长短
            * 特征向量对应椭圆轴的方向
            * 特征值最大的就是第一主成分
        * 详细过程  
            * 标准化的实质就是将矩阵每一列的均值和方差变为1（减去均值，除以标准差），其实就是做了伸缩和平移，于是就将原矩阵Y变换成了X
            * 所以，相关性系数就是R=X*X'（协方差）
            * R是一个实对称矩阵|R-入E|=0(E为单位矩阵)，所以也就会有对应的特征向量和特征值，最终得到就是：
            * 所有主成分对应的特征向量相互是正交的
            * 主成分按特征值大小排序
        * 应用场景
            * 发掘样本之间隐藏的联系
            * 去除样本中冗余的属性和噪声
            * 对数据降维，从而可视化
            * 减少数据，加速学习  


    * 非负矩阵分解(Non-negative Matrix Factorization)  
        * V~=WH, 要求W(n* r), H(r*m)  
        * W是基矩阵，H是系数矩阵，其实可以发现，V中的数字，都是由W中的数字乘上H的一些系数从而得到的
        * 一种方式：从秩分解得到非负分解，其实就是高斯消元法  
        * 基本原理，我们可以改变的是就是上面W和H的r，使得它们可以得到W*H=V  
            * 其中，一般来说，r<=rnak(V),都有
            * 但是很少具备r<=rank(v)+,
            * 这是一种无监督学习方式，缺乏足够的先验知识，因此难以人工标注或进行人工类别标注的成本太高，根据类别未被识别（未标记的）训练样本解决模式识别中的各种问题  
        * 迭代算法：初始的生成一个W，***利用倍增更新规则***，不断地更新W和H  
        * 实际应用：
            * 图像识别；
            * 机器人控制；
            * 时序分割（隐马尔科夫模型，处理时间序列的数据，例如音频视频）；
            * 语音处理；
        * 应用场景：
            * 特征提取：图像特征提取
            * 交通模式提取：道路异常检测中，直接采用交通流量会有以下问题
            * 原始矩阵（交通流量矩阵），表示每一个路段的24小时的车流量，所以就是24列；
            * 分解形式：V=C*P，C（m * r），P（r * h），所以分解出来的模式不能很多
            * 交通模式就是不变的一个情况，基矩阵就是P，研究路段的情况，
            * 你就只需要研究C，C就是他们的系数，他们就是现实交通状况的表现
~~~
下面是使用非负矩阵分解对道路模式进行降维
V=V1;
[n,m]=size(V);
k=min(n,m);%r的数值不能大于矩阵中的最小的行或列
for count = 2:k
    r=count;
    W=rand(n,r);
    H=rand(r,m);
    for loop=1:10%对r目前的数值进行10次迭代，主要是考虑到计算机的运行时间
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

仅仅是上面的代码，我就运行了将近3min，所以，，，我们可以使用matlab现成的函数nnmf来计算
%非负矩阵分解，直接调用nnmf函数
V1=flowdata1(:,4:99);
V2=flowdata2(:,4:99);
V3=flowdata3(:,4:99);


r=3;%设置模式数量
[W1,H1]=nnmf(V1,r);
[W2,H2]=nnmf(V2,r);
[W3,H3]=nnmf(V3,r);
~~~

* 3.机器学习  
    * 主要分为分类（监督学习）和聚类（无监督学习）  
    * 1.分类算法
        * 定义：学习一个函数或模型（分类器），把每个数据项映射到给定类别中的某一个
        * 常见类型：相似度模型；概率模型；线性模型；非线性模型；
        * 应用：银行风险评估；客户类别分类；文本检索；搜索引擎分类；安全入侵检测等
        * 相似度模型：K-近邻（K-nearest neighbours，K-NN），也就是在某个点距离已经标记的点最近，就归为谁：算法现在要使用Kd树，这样就可以使算法复杂度从O（N）降低到O（logN）
        * 线性模型：支持向量机（SVM）
            * 硬间隔
            * 软间隔
            * 非线性分类：核函数
        * 非线性模型：决策树算法（Decision Tree） 优点是容噪，缺点是效率低
            * ID3
            * C4.5
            * CART
        * 概率模型：朴素贝叶斯（Naive Bayes）之所以是朴素贝叶斯主要是因为，它认为所有X的属性全是独立的
            * 已知条件：x={a1,a2,a3,...},对应y的标签{y1，y2，...}
            * 我们知道公式P（B|A）=P(A|B)*P(B)/[E P(A|B) *P(B)]
            * 我们是假设所有的A，即x的属性之间是独立的，所以对于以上公式，我们那只需要计算出P（Ai|B）和P(B)


    * 2.聚类算法
        * 主要用的是K-means；K中心聚类算法；DBSCAN算法
        * 1.K-means
            * 对异常值很敏感，且需要提前确定K
            * 计算时间短，容易解释，聚类效果还不错
        * 2.K中心
            * 过程：先寻找随机寻找几个中心，然后按距离聚类，再根据几个中心计算出新的中心之后再次聚类
            * 它最初随机选择k个对象作为中心点，该算法反复的用非代表对象（非中心点）代替代表对象，试图找出更好的中心点，以改进聚类的质量。 
            * 例子： 空间有5点{A,B,C,D,E}, 各点之间距离关系如表，根据pam算法进行聚类分析。

                * 样本点	A	B	C	D	E     
                * 假设分为2类，以A，B为中心点，初始聚类为{A,C,D}和{B,E}。接下来进行交换（以非代表对象代替代表对象），我们需要计算TCAC、TCAD、TCAE、TCBC、TCBD、TCBE。 
                * TCij表示用非中心点j替换中心点i所产生的代价。 计算TCAC：当A被C替换后，设一指针p遍历所有对象，判断他们是否聚到别的类里。先看A是否变化：C成为中心点后，A离B比A离C近，故A被划分到B簇里。所产生的代价为d(A,B)-d(A,A)=1(d(i,j)表示i划分到中心点j的距离；差值表示属于新的中心点-属于旧的中心点产生的代价。)
                * 看B是否变化：C成为中心点后，B当然离自己是最近的，不变
                * 看C是否变化：C成为中心点后，C划分到C簇里，代价为d(C,C)-d(C,A)=-2
                * 看D是否变化：C成为中心点后，D离C最近，故划分到C里，代价为d（D,C）-d（D,A）=-1；
                * 看E是否变化：C成为中心点后，E离B最近，为0 
                * TCAC就等于上述的代价之和，为1+0-2-1+0=-2。 
                * 同理需要计算TCAD=-2、TCAE=-1、TCBC=-2、TCBD=-2、TCBE=-2 
                * 然后选取代价最小的替换，这里有多个选择，随便选择一个就行。选C的话，新的簇为{C,D}和{A,B,E}。新的簇中心为C，B，继续迭代计算直到收敛。



* 4.异常检测与完工系统 
~~~
%判断经纬度有没有超过北京范围
cnt=0;
for i=1:137085
    if (data(i,5)>=426516480&&data(i,5)<=432783360&&data(i,6)>=145244160&&data(i,6)<=153354240)==0
        cnt=cnt+1;
        problem(cnt)=i;
    end
end

cnt=0;
%判断速度有没有超过90km/h，若不超过，就复制到新的矩阵，否则直接扔掉
for i=1:137085
    if data(i,7)>90
        cnt=cnt+1;
        problemvelocity(cnt)=i;      
    end
end

%客人下车0和客人上车1时速度必须为0
cnt=0;
for i=1:137085
    if (data(i,10)==0 || data(i,10)==1) && data(i,7)~=0
        cnt=cnt+1;
        problemstate(cnt)=i;
    end
end

%如果两个相邻点的距离大于2km，我们认为不真实
cnt=0;
len=137085;
% for i=1:137084
%     x1=data(i,3)*111;
%     x2=data(i+1,3)*111;
%     y1=data(i,4)*111;
%     y2=data(i+1,4)*111;
%     distance=sqrt((x1-x2).^2+(y1-y2).^2);
%     if (distance>=2 && data(i,1)==data(i+1,1))
%         cnt=cnt+1;
%         problemdis(cnt)=i;
%     end   
% end  
for i=1:137084
    LonA=data(i,3);   
    LatA=data(i,4);
    LonB=data(i+1,3);
    LatB=data(i+1,4);
    dis1=dis(LonA,LatA,LonB,LatB);
    if (dis1>=2 && data(i,1)==data(i+1,1))
        cnt=cnt+1;
        problemdis(cnt)=i;
    end   
end  

% %停住点：我们认为在30分钟内没有50m的距离的轨迹为无效轨迹
% cnt=0;
% len=137085;
% i=1;
% j=1;
% while(1)
%     while(i+j<=137085&&data(i+j,2)-data(i,2)<30*60&&data(i+j,1)==data(i,1))          %两次测量时间差小于30分钟
%         LonA=data(i,3);   
%         LatA=data(i,4);
%         LonB=data(i+j,3);
%         LatB=data(i+j,4);
%         dis1=dis(LonA,LatA,LonB,LatB);
%         if(cnt~=0&&dis1<0.05&&i+j>problemstop(cnt))                           %两次测量距离差小于0.05km
%            cnt=cnt+1;
%            problemstop(cnt)=i+j; 
%         end
%         if(cnt==0&&dis1<0.05)
%            cnt=cnt+1;
%            problemstop(cnt)=i+j;
%         end
% 
%         j=j+1;
%     end
%     i=i+j;
%     j=1;
%     if(i>=137085)
%         break;
%     end
% end

j=0;
problemlongtime=[];
flag=0
for i=1:137084
    i
    if data(i,1)~=data(i+1,1);
        continue;
    end
    if data(i,9)~=0 && data(i,9)~=1 
        continue;
    end
    for k=1:137085-i
        datediff=((fix((mod(data(i,2),100000000)-mod(data(i+k,2),100000000))/1000000)*24+fix(mod(data(i,2),1000000)-mod(data(i+k,2),1000000))/10000)*60+fix(mod(data(i,2),10000)-mod(data(i+k,2),10000))/100)*60+mod(data(i,2),100)-mod(data(i+k,2),100);
        x1=data(i,3)*111;
        x2=data(i+k,3)*111;
        y1=data(i,4)*111;
        y2=data(i+k,4)*111;
        distan=sqrt((x1-x2).^2+(y1-y2).^2);
        if distan>=0.5
            break
            continue
        end
        if datediff<=-1800
            distance=6371004*acos(1-(power((sin((90-data(i,3))*pi/180)*cos(data(i,2)*pi/180)-sin((90-data(i+k,3))*pi/180)*cos(data(i+k,2)*pi/180)),2)+power((sin((90-data(i,3))*pi/180)*sin(data(i,2)*pi/180)-sin((90-data(i+k,3))*pi/180)*sin(data(i+k,2)*pi/180)),2)+power((cos((90-data(i,3))*pi/180)-cos((90-data(i+k,3))*pi/180)),2))/2);
            break
        end
    end
    if distan<=0.05
        if flag~=1
            problemlongtime=[problemlongtime;i,i+k];
            temp=i+k;
        end
        flag=1;
    end
    if distan<=0.05
        if i<temp
            continue
        end
        problemlongtime=[problemlongtime;i,i+k];
        temp=i+k;
        j=j+1;
        j
    end
end
    problemlongtime=problemlongtime';





%非负矩阵分解，直接调用nnmf函数
V1=flowdata1(:,4:99);
V2=flowdata2(:,4:99);
V3=flowdata3(:,4:99);

r=3;%设置模式数量
[W1,H1]=nnmf(V1,r);
[W2,H2]=nnmf(V2,r);
[W3,H3]=nnmf(V3,r);

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

~~~

* 5.模型评价
    * 混淆矩阵：P-R曲线
    * 多混淆矩阵：
        * 先分别计算查准率和查全率，再平均
        * ROC与AUC
            * 接收者工作特征曲线（Receiver Operating Characteristic）
            * AUC（Area Under ROC curve）
            * [ROC](https://baike.baidu.com/pic/ROC%E6%9B%B2%E7%BA%BF/775606/0/fc5e5f34f075091a241f1470?fr=lemma&ct=single#aid=0&pic=fc5e5f34f075091a241f1470)
            
