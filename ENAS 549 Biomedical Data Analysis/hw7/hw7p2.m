%% Problem 2

y0 = [1000; 1; 0];
[t, y] = ode45(@epidemic_model, [0 10], y0);

% find the week that the disease subside
max_index = find(y(:,2)==max(y(:,2)));
for i=max_index:1:size(t,1)
    if 0.005*y0(1) > y(i, 2)
        t_subside = t(i);
        break
    end
end

figure;
plot(t, y(:,1));
hold on
plot(t, y(:,2));
hold on
plot(t, y(:,3));
legend('S', 'I', 'R')

function dydt = epidemic_model(t, y)

% y(1) = S
% y(2) = I
% y(3) = R

alpha = 0.005;
beta = 1;

dydt = [-alpha*y(1)*y(2); alpha*y(1)*y(2) - beta*y(2); beta*y(2)];

end