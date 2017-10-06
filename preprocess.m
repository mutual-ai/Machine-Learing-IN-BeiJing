clc;
clear all;

a=xlsread('D:\����\2017�����п�Ժ�����ھ�\code\all-group.xlsx');
data=a(1:137085,3:12);

%�жϾ�γ����û�г���������Χ
cnt=0;
for i=1:137085
    if (data(i,5)>=426516480&&data(i,5)<=432783360&&data(i,6)>=145244160&&data(i,6)<=153354240)==0
        cnt=cnt+1;
        problem(cnt)=i;
    end
end

cnt=0;
%�ж��ٶ���û�г���90km/h�������������͸��Ƶ��µľ��󣬷���ֱ���ӵ�
for i=1:137085
    if data(i,7)>90
        cnt=cnt+1;
        problemvelocity(cnt)=i;      
    end
end

%�����³�0�Ϳ����ϳ�1ʱ�ٶȱ���Ϊ0
cnt=0;
for i=1:137085
    if (data(i,10)==0 || data(i,10)==1) && data(i,7)~=0
        cnt=cnt+1;
        problemstate(cnt)=i;
    end
end

%����������ڵ�ľ������2km��������Ϊ����ʵ
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

% %ͣס�㣺������Ϊ��30������û��50m�ľ���Ĺ켣Ϊ��Ч�켣
% cnt=0;
% len=137085;
% i=1;
% j=1;
% while(1)
%     while(i+j<=137085&&data(i+j,2)-data(i,2)<30*60&&data(i+j,1)==data(i,1))          %���β���ʱ���С��30����
%         LonA=data(i,3);   
%         LatA=data(i,4);
%         LonB=data(i+j,3);
%         LatB=data(i+j,4);
%         dis1=dis(LonA,LatA,LonB,LatB);
%         if(cnt~=0&&dis1<0.05&&i+j>problemstop(cnt))                           %���β��������С��0.05km
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



