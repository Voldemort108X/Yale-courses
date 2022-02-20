
%% p2 part (a)
for i=1:100 
    tt(i)=2*i+3; 
    yy(i)=5*tt(i)*log(1+tt(i)); 
end

i = 1:1:100;
t = 2*i+3;
y = 5*t.*log(1+t);

%% p2 part (b)
m=y(1); 
for k=2:length(y) 
   if y(k) < m 
        m=y(k); 
    end 
end 

mm = min(y);

%% p2 part (c)
s=0;
k=0; 
for i=1:2:length(y) 
    k=k+1; 
    s=s+k*y(i); 
end 

i = 1:2:length(y);
kk = 1:1:length(i);
ss = sum(kk.*y(i));

%% p3 part(b), (c)
n = 1;
tol1 = 1e-4;
tol2 = 1e-5;
while abs(pi-approx_pi(n)) > tol2
    n = n+1;
end

%% p4 part (a)
n = 1;
k = 50;
while sum_abs_sine(n) < k
    n = n+1;
end
n = n-1;

%% p6
t = 0:0.1:15;
% a = 2;
a = 2.1;
x_t = sin(t);
y_t = cos(a*t);

figure;
plot(t, x_t ,'b');
hold on
plot(t, y_t, 'r');
legend('x(t)','y(t)');
xlabel('t');
ylabel('x(t) or y(t)');
% title('a = 2');
title('a = 2.1');

figure;
plot(x_t, y_t);
xlabel('x(t)');
ylabel('y(t)');
% title('a = 2');
title('a = 2.1');