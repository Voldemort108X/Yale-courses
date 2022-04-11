load('fit_2d_gaussian.mat');
figure;
mesh(x,y,z)
title('2D mesh graph of z versus x and y');

% xdata = [x;y];
A0 = max(max(z));
sigma_u0 = 2;
sigma_v0 = 2;
[x0, y0] = find(z == A0);
theta0 = 0;
beta0 = [A0; x0; y0; sigma_u0; sigma_v0; theta0];
[beta, ss, residuals] = lsqcurvefit(@fit_2d_gauss, beta0, x, z, [], [], [], y);

% plot residuls
figure;
mesh(x, y, residuals);
title('Residuals');

% plot data fits as a function of y
eta = fit_2d_gauss(beta, x, y);
[center_x, center_y] = find(eta == max(max(eta)));
eta_y = eta(center_x,:);
z_y = z(center_x,:);
y_slice = y(:,center_x)';
figure;
plot(y_slice, eta_y);
hold on
plot(y_slice, z_y);
title('Data fits as a function of y');
legend('fits','data');

% plot data fits as a function of x
eta_x = eta(:, center_y);
z_x = z(:, center_y);
x_slice = x(center_y, :)';
figure;
plot(x_slice, eta_x);
hold on
plot(x_slice, z_x);
title('Data fits as a function of x');
legend('fits','data');


function eta=fit_2d_gauss(beta, x, y)
%     x = xdata(1:31,:);
%     y = xdata(32:62,:);
    A = beta(1);
    x0 = beta(2);
    y0 = beta(3);
    sigma_u = beta(4);
    sigma_v = beta(5);
    theta = beta(6);
    u = (x-x0)*cos(theta) - (y-y0)*sin(theta);
    v = (x-x0)*sin(theta) + (y-y0)*cos(theta);
    eta = A*exp(-u.^2./(2*sigma_u.^2)-v.^2./(2*sigma_v.^2));
end
