%% Problem 2 part 1
mu = 1;
Q0 = @(x) 0.5 * (1 - erf(x/sqrt(2)));
Q1 = @(x) 0.5 * (1 - erf((x-mu)/sqrt(2)));

gamma = linspace(-2, 4, 1000);
alpha = Q0(gamma);
beta = Q1(gamma);

Area = abs(trapz(alpha, beta));
NM_region= 2*(Area-0.5);

plot(alpha, beta, 'b');
hold on
plot(1-alpha, 1-beta, 'b');
xlabel('alpha');
ylabel('beta');

%% Problem 2 part 2
cnt = 1;
mu = 1;

for n = [5, 10, 15]
    Q0 = @(x) 0.5 * (1 - erf(x/sqrt(2/n)));
    Q1 = @(x) 0.5 * (1 - erf((x-mu)/sqrt(2/n)));

    gamma = linspace(-2, 4, 1000);
    alpha = Q0(gamma);
    beta = Q1(gamma);

    Area = abs(trapz(alpha, beta));
    NM_region(cnt) = 2*(Area-0.5);

    figure;
    plot(alpha, beta, 'b');
    hold on
    plot(1-alpha, 1-beta, 'b');
    xlabel('alpha');
    ylabel('beta');
    title(['n=',num2str(n)])
    
    cnt = cnt + 1;
end

%% Problem 3 part 2
mu = 1;
theta = linspace(0,1,1000);

psi_theta = ((mu^2+theta.^2)./(2*mu)).^2;
E0 = psi_theta;
E1 = psi_theta - theta;
plot(E0, E1);