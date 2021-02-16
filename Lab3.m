%1
clear all;
close all;
n=100;
ds=4;%�������� ������������ ���� ���������
P1=randn(2,n);
P2=randn(2,n)+ds;
a1(1,1:n)=ds;
a1(2,1:n)=0;
P3=randn(2,n)+a1;
a2(1,1:n)=0;
a2(2,1:n)=ds;
P4=randn(2,n)+a2;
P=[P1 P2 P3 P4];%������� ������
T1(1:2*n)=1;
T2(1:2*n)=0;
T=[T1 T2]; %������� ������
figure('Name','�������� ����������� ��������');%����������� ����
plotpv(P,T);%����������� ��������

%�������� � �������� ��
toutn=0.1+(0.9-0.1)*T;%������� �����
bpn=feedforwardnet([5 1],'trainrp');
view(bpn);
bpn=init(bpn);
bpn.trainParam.epochs=1000;
bpn=train(bpn,P,toutn)

n1=100;
P1=randn(2,n1);
P2=randn(2,n1)+ds;
P=[P1 P2];%������� ������
a=sim(bpn,P);%������������� ��������� ����
for i=1:2*n1,
    if a(i)>=0.5 a(i)=1; else a(i)=0; end; % ���������� ���������� �����������
end;
figure('Name','������ ������� ����');%����������� ����;
plotpv(P,a);%����������� ��������
OC1=1-sum(a)/(2*n1) ;
disp(OC1);

a1(1,1:n1)=ds;
a1(2,1:n1)=0;
P3=randn(2,n1)+a1;
a2(1,1:n1)=0;
a2(2,1:n1)=ds;
P4=randn(2,n1)+a2;
P=[P3 P4];
b=sim(bpn,P);
for i=1:2*n1,
    if b(i)>0.5 b(i)=1; else b(i)=0; end; 
end;
figure('Name','������ ������� ����');%����������� ����;
plotpv(P,b);
OC2=sum(b)/(2*n1);
disp(OC2);
