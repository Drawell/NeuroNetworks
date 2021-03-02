clear all;
close all;
n=40;%���������� ������� ��������� �������� � ������ ������ �������
%P1=randn(2, n);
P1x=normrnd(0, 0.4, [1,n]);
P1y=normrnd(0, 1.8, [1,n]);
P1 = [P1x; P1y];
P12x=normrnd(5, 1.8, [1,n]);
P12y=normrnd(0, 0.4, [1,n]);
P12 = [P12x; P12y];

%�����������  ��������������� �������� ��� �������� ������� ������
P2x=normrnd(10, 0.4, [1,n]);
P2y=normrnd(0, 1.8, [1,n]);
P2=[P2x; P2y];
P=[P1 P12 P2];%������� ������� ��������
T1(1:n)=1;
T12(1:n)=1;
T2(1:n)=0;
T=[T1 T12 T2];%������� �������� (�������) ��������
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


n1=100;
P41x=normrnd(0, 0.4, [1,n1]);
P41y=normrnd(0, 1.8, [1,n1]);
P41 = [P41x; P41y];

P42x=normrnd(5, 1.8, [1,n1]);
P42y=normrnd(0, 0.4, [1,n1]);
P42 = [P42x; P42y];

P4=[P41 P42];

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
OC1=1-sum(a)/(2*n1);
disp(OC1);

P5x=normrnd(10, 0.4, [1,n1]);
P5y=normrnd(0, 1.8, [1,n1]);
P5=[P5x; P5y];

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

