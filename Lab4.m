%1 - исходные данные
clear all;
close all
mi=-10; ma=10; %диапазон значений
delt=5;%шаг дискретизации
intrvl=mi:delt:ma;
cnt=1;
for x1=intrvl,
       for x2=mi:delt:ma,
           for x3=mi:delt:ma,
               A(cnt)=-(x1+x2+x3);
               B(cnt)=x2*x3+x1*x3+x2*x1;
               C(cnt)=-x1*x2*x3;
               X(:,cnt)=[x1; x2; x3];
               cnt=cnt+1;
          end;
      end;
end;
P=[A;B;C];

%2 - исходные данные
clear all;
close all;
mi=-10; ma=10; %диапазон значений
delt=5;%шаг дискретизации
intrvl=mi:delt:ma;
cnt=1;
for x1=intrvl,
       for x2=x1:delt:ma,
           for x3=x2:delt:ma,
               A(cnt)=-(x1+x2+x3);
               B(cnt)=x2*x3+x1*x3+x2*x1;
               C(cnt)=-x1*x2*x3;
               X(:,cnt)=[x1; x2; x3];
               cnt=cnt+1;
          end;
      end;
end;
P=[A;B;C];


%3 - создание и обучение Ќ—
Xn=(X-mi)*0.8/(ma-mi)+0.1;
[Pn,Pmin,Pmax]=premnmx(P);
bpn=feedforwardnet([200 3],'trainrp');
view(bpn);
bpn=init(bpn);
bpn.trainParam.epochs=500;
bpn=train(bpn,Pn,Xn);

%4 тестирование Ќ—
kk=10;
for cnt=1:kk,
x1=mi+rand*(ma-mi);
x2=x1+rand*(ma-x1);
x3=x2+rand*(ma-x2);
A1(cnt)=-(x1+x2+x3);
B1(cnt)=x2*x3+x1*x3+x2*x1;
C1(cnt)=-x1*x2*x3;
X1(:,cnt)=[x1; x2; x3];
end;
P1=[A1;B1;C1];
P1n=tramnmx(P1,Pmin,Pmax);
X1n=(X1-mi)*0.8/(ma-mi)+0.1;
y=sim(bpn,P1n);
mse(X1n-y)

