clear all;
close all;
Isize=5;%размер обрабатываемых изображений
IA=[0 0 1 0 0; %условное бинарное отображение буквы A
    0 1 0 1 0;
    0 1 0 1 0;
    1 1 1 1 1;
    1 0 0 0 1];

figure('Name','Буква - A - исходное');
imagesc(IA);
IB=[0 1 1 1 0; %условное бинарное отображение буквы B
    0 1 0 1 0;
    0 1 1 0 0;
    0 1 0 1 0;
    0 1 1 1 0];
figure('Name','Буква - B - исходное');
imagesc(IB);
IC=[0 1 1 1 0; %условное бинарное отображение буквы С
       0 1 0 0 0;
       0 1 0 0 0;
       0 1 0 0 0;
       0 1 1 1 0];
figure('Name','Буква - C - исходное');
imagesc(IC);
for i=1:Isize,
      for j=1:Isize,
          if IA(i,j)==0 IA(i,j)=-1; end;
          if IB(i,j)==0 IB(i,j)=-1; end;
          if IC(i,j)==0 IC(i,j)=-1; end;
      end;
end;
%развертка матриц в векторы
cA=IA(:);
cB=IB(:);
cC=IC(:);
T=[cA cB cC]
net=newhop(T);%создание сети  Хопфилда

po=0.1;%вероятность искажения единичного элемента (пикселя) изображения
IA1=IA;
for i=1:Isize,
      for j=1:Isize,
          x=rand;
          if x<po IA1(i,j)=-IA(i,j); end
     end;
end;
cA1=IA1(:);
Ai=cA1;
figure('Name','Буква - A - испорченное');
imagesc(reshape(Ai,[5 5]));

% восстановление изображения
[Y,Pf,Af]=sim(net, 1, [], Ai);
Yc=round(Y)
figure('Name','Буква - A - восстановленное');
imagesc(reshape(Yc,[5 5]));

figure('Name','Буква - A - результат');
subplot(1,3,1);
imagesc(IA);
subplot(1,3,2);
imagesc(reshape(Ai,[5 5]));
subplot(1,3,3);
imagesc(reshape(Yc,[5 5]));

