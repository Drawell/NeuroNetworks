%1
clear all;
close all;
P1=[-0.5 -0.5  0.4  -0.2;
   -0.5  0.5 -0.4   1.0]; % 
%�������  (�����������) ������
T1=[1 1 0 0];
figure ('Name','������� �������')
plotpv(P1,T1);

%�������� � �������� ��
net = perceptron;
net=init(net);
net.trainParam.epochs=20;
net=train(net,P1,T1);
%��������� ����������� ��������� linehanle ��� ����������� ����������� 
%�������, ����������� ��������� ����� ����� ��������
linehandle=plotpc(net.IW{1},net.b{1});
%����������� ����������� �����
plotpc(net.IW{1},net.b{1});
%������������� ������ �������
P=[-0.6;0.5];
a=sim(net,P);
figure ('Name','������������� ������ �������')
plotpv(P,a);
ThePoint=findobj(gca,'type','line');%������� ���� ��� ������ �������
set(ThePoint, 'Color','red');
hold on;%��������� ������ ���������� �������� � ����������� ����
plotpv(P1,T1);
plotpc(net.IW{1},net.b{1});
hold off;%���������� ������ ���������� �������� � ����������� ����

%2
clear all;
close all;
n=20;%���������� ������� ��������� �������� � ������ ������ �������
P1=randn(2,n);
%�����������  ��������������� �������� ��� �������� ������� ������
A(2,1:n)=4;%������ �������� ������������ ��� OX
A(1,1:n)=4;%������ �������� ������������ ��� OY
P2=randn(2,n);
P=[P1 P2+A];%������� ������� ��������
T1(1:n)=1;
T2(1:n)=0;
T=[T1 T2];%������� �������� (�������) ��������
%���������� ���������� ���������� �������� ������� � ������� �������
figure ('Name','����������� ����������� �����')
plotpv(P,T);
net=perceptron;
net=init(net);
E=1;% ������ ��������� �������� ������
linehandle=plotpc(net.IW{1},net.b{1});
%���������� ����� ���� E �� ����� 0
while(mse(E))',
       [net,Y,E]=adapt(net,P,T);
       linehandle=plotpc(net.IW{1},net.b{1},linehandle);
       plotpc(net.IW{1},net.b{1});
       drawnow;%������� ���� ��������
end;
P3=[0.6;1.1];
a=sim(net,P3);
figure ('Name','������������� ������ �������')
plotpv(P3,a);
ThePoint=findobj(gca,'type','line');%������� ���� ��� ������ �������
set(ThePoint, 'Color','red');
hold on;%��������� ������ ���������� �������� � ����������� ����
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;

n1=100;
P4=randn(2,n1)
a=sim(net,P4);
figure('Name','������ ������� ����');%����� ����������� ����
plotpv(P4,a);
ThePoint=findobj(gca,'type','line');
set(ThePoint, 'Color','red');
hold on;%��������� ������ ���������� ��������
%� ����������� ����
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;%���������� ������ ���������� ��������
%� ����������� ����
OC1=1-sum(a)/n1;
disp(OC1);

A1(1,1:n1)=3;
A1(2,1:n1)=3;
P5=randn(2,n1)+A1;
a11=sim(net,P5);
%��������� ������ ���������� ��������
figure('Name','������ ������� ����');%����� ����������� ����
hold on;%��������� ������ ���������� ��������
plotpv(P5,a11);
ThePoint=findobj(gca,'type','line');
set(ThePoint, 'Color','red');
hold on;
%� ����������� ����
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;%���������� ������ ���������� ��������
%� ����������� ����
OC2=sum(a11)/n1;
disp(OC2);

