%% problem 4
kd_h = 10;
kd_l = 0.1;
f_H = 0.8;
B_max = 25;

B = 1:1:100;
for idx = 1:1:size(B,2)
    fun = @(F)(F*f_H*B_max)/(F+kd_h) + (F*(1-f_H)*B_max)/(F+kd_l) - B(idx);
    F(idx) = fsolve(fun, 0);
end
plot(B, B./F);
hold on
B_H = 1:1:round(f_H*B_max);
plot(B_H, (f_H*B_max-B_H)./kd_h);
hold on
B_L = 1:1:round((1-f_H)*B_max);
plot(B_L, ((1-f_H)*B_max-B_L)./kd_l);
legend('B/F', 'B_H/F','B_L/F');
xlabel('B');
ylabel('B/F');