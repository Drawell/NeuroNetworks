clc;
%or
clear all;
close all;
net=newp([0 1;-2 2],1);
net=init(net);
%матрица входных векторов
P1=[0 0 1 1;
    0 1 0 1];
% требуемые целевые выходы
T1=[0 1 1 1]; 
net.trainParam.epochs=5;
net=train(net,P1,T1);
y1=sim(net,P1);
disp(y1);
%тестирование
%матрица входных векторов
P2={[0;1] [1;1] [1;0] [0;0]};
%моделирование сети с входным сигналом P2
y2=sim(net,P2);
disp(y2);


% xor
clear all;
close all;
net=newp([0 1;-2 2],1);
net=init(net);
%матрица входных векторов
P1=[0 0 1 1;
    0 1 0 1];
% требуемые целевые выходы
T1=[0 1 1 0]; 
net.trainParam.epochs=10;
net=train(net,P1,T1);
y1=sim(net,P1);
disp(y1);
%тестирование
%матрица входных векторов
P2={[0;1] [1;1] [1;0] [0;0]};
%моделирование сети с входным сигналом P2
y2=sim(net,P2);
disp(y2);


n = -1:0.01:1;
a = 0.5;
b = n ./ (a + abs(n)); 
figure;
plot(n,b);


