%1
clear all;
close all;
P1=[-0.5 -0.5  0.4  -0.2;
   -0.5  0.5 -0.4   1.0]; % 
%целевой  (маркирующий) вектор
T1=[1 1 0 0];
figure ('Name','÷елевые вектора')
plotpv(P1,T1);

%создание и обучение Ќ—
net = perceptron;
net=init(net);
net.trainParam.epochs=20;
net=train(net,P1,T1);
%получение управл€ющей структуры linehanle дл€ изображени€ раздел€ющей 
%границы, формируемой нейронной сетью после обучени€
linehandle=plotpc(net.IW{1},net.b{1});
%изображение раздел€ющей линии
plotpc(net.IW{1},net.b{1});
%классификаци€ нового вектора
P=[-0.6;0.5];
a=sim(net,P);
figure ('Name',' лассификаци€ нового вектора')
plotpv(P,a);
ThePoint=findobj(gca,'type','line');%зададим цвет дл€ нового маркера
set(ThePoint, 'Color','red');
hold on;%включение режима добавлени€ графиков в графическом окне
plotpv(P1,T1);
plotpc(net.IW{1},net.b{1});
hold off;%отключение режима добавлени€ графиков в графическом окне

%2
clear all;
close all;
n=20;%количество входных двумерных векторов в каждом классе образов
P1=randn(2,n);
%локализаци€  математического ожидани€ дл€ векторов второго класса
A(2,1:n)=4;%задаем смещение относительно оси OX
A(1,1:n)=4;%задаем смещение относительно оси OY
P2=randn(2,n);
P=[P1 P2+A];%матрица входных векторов
T1(1:n)=1;
T2(1:n)=0;
T=[T1 T2];%матрица выходных (целевых) векторов
%графически отображаем координаты векторов первого и второго классов
figure ('Name','ќтображение раздел€ющей линии')
plotpv(P,T);
net=perceptron;
net=init(net);
E=1;% задаем пороговое значение ошибки
linehandle=plotpc(net.IW{1},net.b{1});
%органиаци€ цикла пока E не равно 0
while(mse(E))',
       [net,Y,E]=adapt(net,P,T);
       linehandle=plotpc(net.IW{1},net.b{1},linehandle);
       plotpc(net.IW{1},net.b{1});
       drawnow;%очистка окна графиков
end;
P3=[0.6;1.1];
a=sim(net,P3);
figure ('Name',' лассификаци€ нового вектора')
plotpv(P3,a);
ThePoint=findobj(gca,'type','line');%зададим цвет дл€ нового маркера
set(ThePoint, 'Color','red');
hold on;%включение режима добавлени€ графиков в графическом окне
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;

n1=100;
P4=randn(2,n1)
a=sim(net,P4);
figure('Name','ќшибка первого рода');%новое графическое окно
plotpv(P4,a);
ThePoint=findobj(gca,'type','line');
set(ThePoint, 'Color','red');
hold on;%включение режима добавлени€ графиков
%в графическом окне
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;%отключение режима добавлени€ графиков
%в графическом окне
OC1=1-sum(a)/n1;
disp(OC1);

A1(1,1:n1)=3;
A1(2,1:n1)=3;
P5=randn(2,n1)+A1;
a11=sim(net,P5);
%включение режима добавлени€ графиков
figure('Name','ќшибка второго рода');%новое графическое окно
hold on;%включение режима добавлени€ графиков
plotpv(P5,a11);
ThePoint=findobj(gca,'type','line');
set(ThePoint, 'Color','red');
hold on;
%в графическом окне
plotpv(P,T);
plotpc(net.IW{1},net.b{1});
hold off;%отключение режима добавлени€ графиков
%в графическом окне
OC2=sum(a11)/n1;
disp(OC2);

