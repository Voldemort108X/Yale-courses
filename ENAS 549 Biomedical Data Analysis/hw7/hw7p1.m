%% Problem 1

% part 1
c0 = [151.52; 150];
opt = odeset('Events', @Event_parta);
[t_p1, c_p1] = ode45(@urea_model_p1, [0 50], c0, opt);

figure;
plot(t_p1, c_p1(:,1));
hold on
plot(t_p1, c_p1(:,2));
legend('C1','C2');

% part 2
c0_p2 = [83, 75];
opt_p2 = odeset('Events', @Event_partb);
[t_p2, c_p2] = ode45(@urea_model_p2, [0 50], c0_p2, opt_p2);
    
figure;
plot(t_p2, c_p2(:,1));
hold on
plot(t_p2, c_p2(:,2));
legend('C1','C2');

% part 3


function dcdt = urea_model_p1(t, c)
R = 100; 
k12 = 33;
k21 = 33;
V1 = 10;
V2 = 25;
k2 = 8;

dcdt = [(R - k12*c(1) + k21*c(2))/V1; (k12*c(1) - k21*c(2) - k2*c(2))/V2];

end

function [value, isterminal, direction] = Event_parta(t, c)
value      = (c(2) < 76 && c(2) > 74);
isterminal = 1;   % Stop the integration
direction  = 0;
end


function dcdt = urea_model_p2(t, c)
R = 100; 
k12 = 33;
k21 = 33;
V1 = 10;
V2 = 25;
k2 = 0;

dcdt = [(R - k12*c(1) + k21*c(2))/V1; (k12*c(1) - k21*c(2) - k2*c(2))/V2];

end

function [value, isterminal, direction] = Event_partb(t, c)
value      = (c(2) > 149 && c(2) < 151);
isterminal = 1;   % Stop the integration
direction  = 0;
end