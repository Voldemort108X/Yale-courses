%% problem 4
clear all
load('fit_ode_B.mat');
xdata = t;
ydata = y;

kon_init = ((ydata(2)-ydata(1))/(t(2)-t(1)))/(f0*bmax);
koff_init = (ydata(end)-ydata(end-1))/(t(end)-t(end-1))/(bmax);

[x_hat,Rsdnrm,Rsd,ExFlg,OptmInfo,Lmda,Jmat] = lsqcurvefit(@RB_model, [kon_init; koff_init], xdata, ydata);

y0 = [0; f0];
[t_list_init, B_list_init] = ode45(@RB_ode_init, xdata, y0);
[t_list_fit, B_list_fit] = ode45(@RB_ode_fit, xdata, y0);

figure;
plot(xdata, ydata);
hold on
plot(xdata, B_list_init(:,1));
hold on
plot(xdata, B_list_fit(:,1));
legend('Original data','Initial guess fit','Fitted curve');


function B=RB_model(x, t)
    f0 = 100;
    y0 = [0; f0];
    [t_list, B_list] = ode45(@RB_ode, t, y0);
    
    function dydt=RB_ode(t, y)
        % y(1) = B
        % y(2) = F
%         Bmax = 50;
        bmax = 50;
        dydt = [x(1)*(bmax-y(1))*y(2)-x(2)*y(1); -(x(1)*(bmax-y(1))*y(2)-x(2)*y(1))];
    end
    B = B_list(:,1)';
end


function dydt=RB_ode_init(t, y)
        % y(1) = B
        % y(2) = F
   x1 = 0.005544766619132;
   x2 = 0.056828930240176;
   Bmax = 50;
   dydt = [x1*(Bmax-y(1))*y(2)-x2*y(1); -(x1*(Bmax-y(1))*y(2)-x2*y(1))];
end

function dydt=RB_ode_fit(t, y)
        % y(1) = B
        % y(2) = F
   x1 = 0.00469943752378903;
   x2 = 0.048263721378474;
   Bmax = 50;
   dydt = [x1*(Bmax-y(1))*y(2)-x2*y(1); -(x1*(Bmax-y(1))*y(2)-x2*y(1))];
end

% y0 = [0; f0];
% [t_list, B_list] = ode45(@RB_ode, [0 50], y0);
% 
% function dydt = RB_ode(t, y)
%         % y(1) = B
%         % y(2) = F
%         x = [2;2];
%         Bmax = 50;
%         dydt = [x(1)*(Bmax-y(1))*y(2)-x(2)*y(1); -(x(1)*(Bmax-y(1))*y(2)-x(2)*y(1))];
% end
