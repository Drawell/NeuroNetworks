%�������� ���� �������� � ������� �� ����������
%����������� ��������� �������� ��� ������������� c ��������������
%������� rands �� -1 �� 1
%1
clear all;
close all;
 P=rands(2,100);
 figure('Name','�������� �������');
 plot(P(1,:),P(2,:),'*g');
 net=newsom([-1 1; -1 1],[5 6]);
 figure('Name','������� ������������ - �� ��������');
 plotsom(net.IW{1,1},net.layers{1}.distances);

 %2- ������������� ��
net.trainParam.epochs=2;
net=train(net,P);
figure('Name','������� ������������ - 2 �����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=5;
net=train(net,P);
figure('Name','������� ������������ - 5 ����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;
 
net.trainParam.epochs=10;
net=train(net,P);
figure('Name','������� ������������ - 10 ����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
 

%3 - ������������ ����
 Y=sim(net,[0.5;0.3]);
 Y
 plot(0.5,0.3,'*k');
 a=sim(net,P);
 %������ ������������ ������ �� ���������
 figure('Name','������ ������������ ������ �� ���������');
bar(sum(a'));


%4 - ������������� ����������� ��������
clear all;
close all;
n=20;
d=4;
A1(1,1:n)=d;
A1(2,1:n)=d;
A2(1,1:n)=d;
A2(2,1:n)=-d;
A3(1,1:n)=-d;
A3(2,1:n)=-d;
A4(1,1:n)=-d;
A4(2,1:n)=d;
P1=randn(2,n)+A1;
P2=randn(2,n)+A2;
P3=randn(2,n)+A3;
P4=randn(2,n)+A4;
P=[P1 P2 P3 P4];
figure('Name','�������� �������');
plot(P(1,:),P(2,:),'*g');
net=newsom([-10 10;-10 10],[2 2]);
figure('Name','������� ������������ - �� ��������');
plotsom(net.IW{1,1},net.layers{1}.distances);

net.trainParam.epochs=10;
net=train(net,P);
figure('Name','������� ������������ - 5 ����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=100;
net=train(net,P);
figure('Name','������� ������������ - 100 ����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=1000;
net=train(net,P);
figure('Name','������� ������������ - 1000 ����');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');

%������������ ����
Y=sim(net,[0.5;0.3]);
Y
plot(0.5,0.3,'*k');
a=sim(net,P);
%������ ������������ ������ �� ���������
figure('Name','������ ������������ ������ �� ���������');
bar(sum(a'));
