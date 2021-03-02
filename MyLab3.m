%1
clear all;
close all;
n=100;

P11x=normrnd(0, 0.4, [1,n]);
P11y=normrnd(0, 1.8, [1,n]);
P11 = [P11x; P11y];

P12x=normrnd(5, 1.8, [1,n]);
P12y=normrnd(1, 0.4, [1,n]);
P12 = [P12x; P12y];

%локализация  математического ожидания для векторов второго класса
P21x=normrnd(5, 1.8, [1,n]);
P21y=normrnd(-1.5, 0.4, [1,n]);
P21 =[P21x; P21y];

P22x=normrnd(10, 0.4, [1,n]);
P22y=normrnd(0, 1.8, [1,n]);
P22 =[P22x; P22y];

P=[P11 P12 P21 P22];%входной вектор
T1(1:2*n)=1;
T2(1:2*n)=0;
T=[T1 T2]; %целевой вектор
figure('Name','Исходное отображение маркеров');%графическое окно
plotpv(P,T);%отображение маркеров

%создание и обучение НС
toutn=0.1+(0.9-0.1)*T;%целевой выход
bpn=feedforwardnet([5 1],'trainrp');
view(bpn);
bpn=init(bpn);
bpn.trainParam.epochs=1000;
bpn=train(bpn,P,toutn)

n1=100;


P11x=normrnd(0, 0.4, [1,n1]);
P11y=normrnd(0, 1.8, [1,n1]);
P11 = [P11x; P11y];

P12x=normrnd(5, 1.8, [1,n1]);
P12y=normrnd(1, 0.4, [1,n1]);
P12 = [P12x; P12y];

P=[P11 P12];%входной вектор
a=sim(bpn,P);%моделирование нейронной сети
for i=1:2*n1,
    if a(i)>=0.5 a(i)=1; else a(i)=0; end; % округление полученных результатов
end;
figure('Name','Ошибка первого рода');%графическое окно;
plotpv(P,a);%отображение маркеров
OC1=1-sum(a)/(2*n1) ;
disp(OC1);

P21x=normrnd(5, 1.8, [1,n1]);
P21y=normrnd(-1.5, 0.4, [1,n1]);
P21 =[P21x; P21y];

P22x=normrnd(10, 0.4, [1,n1]);
P22y=normrnd(0, 1.8, [1,n1]);
P22 =[P22x; P22y];

P=[P21 P22];
b=sim(bpn,P);
for i=1:2*n1,
    if b(i)>0.5 b(i)=1; else b(i)=0; end; 
end;
figure('Name','Ошибка второго рода');%графическое окно;
plotpv(P,b);
OC2=sum(b)/(2*n1);
disp(OC2);
