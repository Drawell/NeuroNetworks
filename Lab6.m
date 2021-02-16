%Изучение сеть Кохонена и примеры ее применения
%определение множества векторов для кластеризации c использованием
%датчика rands от -1 до 1
%1
clear all;
close all;
 P=rands(2,100);
 figure('Name','Исходные векторы');
 plot(P(1,:),P(2,:),'*g');
 net=newsom([-1 1; -1 1],[5 6]);
 figure('Name','Весовые коэффициенты - до обучения');
 plotsom(net.IW{1,1},net.layers{1}.distances);

 %2- развертывание НС
net.trainParam.epochs=2;
net=train(net,P);
figure('Name','Весовые коэффициенты - 2 эпохи');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=5;
net=train(net,P);
figure('Name','Весовые коэффициенты - 5 эпох');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;
 
net.trainParam.epochs=10;
net=train(net,P);
figure('Name','Весовые коэффициенты - 10 эпох');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
 

%3 - тестирование сети
 Y=sim(net,[0.5;0.3]);
 Y
 plot(0.5,0.3,'*k');
 a=sim(net,P);
 %анализ рапределения данных по кластерам
 figure('Name','Анализ рапределения данных по кластерам');
bar(sum(a'));


%4 - кластеризация гауссовских векторов
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
figure('Name','Исходные векторы');
plot(P(1,:),P(2,:),'*g');
net=newsom([-10 10;-10 10],[2 2]);
figure('Name','Весовые коэффициенты - до обучения');
plotsom(net.IW{1,1},net.layers{1}.distances);

net.trainParam.epochs=10;
net=train(net,P);
figure('Name','Весовые коэффициенты - 5 эпох');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=100;
net=train(net,P);
figure('Name','Весовые коэффициенты - 100 эпох');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');
hold off;

net.trainParam.epochs=1000;
net=train(net,P);
figure('Name','Весовые коэффициенты - 1000 эпох');
plotsom(net.IW{1,1},net.layers{1}.distances);
hold on;
plot(P(1,:),P(2,:),'*g');

%тестирование сети
Y=sim(net,[0.5;0.3]);
Y
plot(0.5,0.3,'*k');
a=sim(net,P);
%анализ рапределения данных по кластерам
figure('Name','Анализ рапределения данных по кластерам');
bar(sum(a'));
