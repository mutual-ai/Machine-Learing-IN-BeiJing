function [A]=sortm(x) % ����x��Ϊ������������Ƕ��������������ΪA,�Ӵ�С������
[row,line]=size(x); %�����������ľ�����к���
count=row*line; %�����ð���㷨Ҫ���ж��ٴδ�ѭ���ıȽ�
for i=1:count %ִ�е�i����ѭ��
    for j=1:count-i %����i��ѭ���󣬹���i������������ȥ��������󼸸������Բ�ѭ��
        lie1=floor(j/row)+1; %�����������ڵ�ѭ��ִ�е����ھ������һ��
        hang1=mod(j,row);%�����������ڵ�ѭ��ִ�е����ھ������һ��
        k=j+1;%������������ѭ��ִ�е�������һ��Ҫ���Ƚϵ����ھ������һ�к���
        lie2=floor(k/row)+1;
        hang2=mod(k,row);
        if(hang1==0) %�����������˵�������һ�У�����������j����row���̣����ü�1
            hang1=row;
            lie1=floor(j/row);
        end
        if(hang2==0) %�����������˵�������һ�У�����������k����row���̣����ü�1
            hang2=row;
            lie2=floor(k/row);
        end
        if(x(hang2,lie2)>=x(hang1,lie1))%����Ƚ������������һ�����������ǾͰ����û���ǰ��ȥ
            temp=x(hang2,lie2);
            x(hang2,lie2)=x(hang1,lie1);
            x(hang1,lie1)=temp;
        end
    end
end
A=x;%���ս�A���


