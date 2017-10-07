function [A]=sortm(x) % 矩阵x作为输入变量，我们对其进行排序后，输出为A,从大到小的排列
[row,line]=size(x); %计算这个输入的矩阵的行和列
count=row*line; %计算出冒泡算法要进行多少次大循环的比较
for i=1:count %执行第i次外循环
    for j=1:count-i %经过i次循环后，共有i个数被沉淀下去，所以最后几个数可以不循环
        lie1=floor(j/row)+1; %计算我们现在的循环执行的数在矩阵的哪一列
        hang1=mod(j,row);%计算我们现在的循环执行的数在矩阵的哪一列
        k=j+1;%计算我们现在循环执行的数的下一个要来比较的数在矩阵的哪一行和列
        lie2=floor(k/row)+1;
        hang2=mod(k,row);
        if(hang1==0) %如果被整除，说明在最后一行，且列数就是j除以row的商，不用加1
            hang1=row;
            lie1=floor(j/row);
        end
        if(hang2==0) %如果被整除，说明在最后一行，且列数就是k除以row的商，不用加1
            hang2=row;
            lie2=floor(k/row);
        end
        if(x(hang2,lie2)>=x(hang1,lie1))%如果比较起来，后面的一个数更大，我们就把它置换到前面去
            temp=x(hang2,lie2);
            x(hang2,lie2)=x(hang1,lie1);
            x(hang1,lie1)=temp;
        end
    end
end
A=x;%最终将A输出


