% %1
% clc;
% clear all;
% close all;
% %создадим персептрон с двуэлементным входом и одним нейроном; диапазон
% %значений входа - [0 1] и [-2 2].
% %1.1
% net=newp([0 1;-2 2],1)
% %1.2
% net = perceptron
% 
% %2
% clear all;
% net=newp([0 1;-2 2],1);
% %инициализация
% net=init(net);
% %входные векторы
% P1={[0;0] [0;1] [1;0] [1;1]};
% y1=sim(net,P1);
% disp(y1);


%3 - train
clear all;
close all;
net=newp([0 1;-2 2],1);
net=init(net);
%матрица входных векторов
P1=[0 0 1 1;
    0 1 0 1];
% требуемые целевые выходы
T1=[0 0 0 1]; 
net.trainParam.epochs=5;
net=train(net,P1,T1);
y1=sim(net,P1);
disp(y1);
%тестирование
%матрица входных векторов
P2={[0;1] [1;1] [1;0] [0;1]};
%моделирование сети с входным сигналом P2
y2=sim(net,P2);
disp(y2);


% 4
hardlim
n = -2:0.01:2;
b = hardlim(n);
figure;
plot(n,b);

% logsig
n = -2:0.01:2;
b = logsig(n); 
figure;
plot(n,b);

%tansig
n = -2:0.01:2;
b = tansig(n);
figure;
plot(n,b);

%purelin 
n =-2:0.01:2;
b = purelin(n);
figure;
plot(n,b);

