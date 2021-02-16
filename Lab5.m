clear all;
close all;
Isize=5;%������ �������������� �����������
IA=[0 0 1 0 0; %�������� �������� ����������� ����� A
    0 1 0 1 0;
    0 1 0 1 0;
    1 1 1 1 1;
    1 0 0 0 1];

figure('Name','����� - A - ��������');
imagesc(IA);
IB=[0 1 1 1 0; %�������� �������� ����������� ����� B
    0 1 0 1 0;
    0 1 1 0 0;
    0 1 0 1 0;
    0 1 1 1 0];
figure('Name','����� - B - ��������');
imagesc(IB);
IC=[0 1 1 1 0; %�������� �������� ����������� ����� �
       0 1 0 0 0;
       0 1 0 0 0;
       0 1 0 0 0;
       0 1 1 1 0];
figure('Name','����� - C - ��������');
imagesc(IC);
for i=1:Isize,
      for j=1:Isize,
          if IA(i,j)==0 IA(i,j)=-1; end;
          if IB(i,j)==0 IB(i,j)=-1; end;
          if IC(i,j)==0 IC(i,j)=-1; end;
      end;
end;
%��������� ������ � �������
cA=IA(:);
cB=IB(:);
cC=IC(:);
T=[cA cB cC]
net=newhop(T);%�������� ����  ��������

po=0.1;%����������� ��������� ���������� �������� (�������) �����������
IA1=IA;
for i=1:Isize,
      for j=1:Isize,
          x=rand;
          if x<po IA1(i,j)=-IA(i,j); end
     end;
end;
cA1=IA1(:);
Ai=cA1;
figure('Name','����� - A - �����������');
imagesc(reshape(Ai,[5 5]));

% �������������� �����������
[Y,Pf,Af]=sim(net, 1, [], Ai);
Yc=round(Y)
figure('Name','����� - A - ���������������');
imagesc(reshape(Yc,[5 5]));

figure('Name','����� - A - ���������');
subplot(1,3,1);
imagesc(IA);
subplot(1,3,2);
imagesc(reshape(Ai,[5 5]));
subplot(1,3,3);
imagesc(reshape(Yc,[5 5]));

