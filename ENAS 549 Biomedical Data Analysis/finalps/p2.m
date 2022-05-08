%% Problem 2
clear all; close all;

k01 = 1;
k21 = 0.1;
k12 = 0.1;
k32_list = 0.05:0.05:0.4;
tspan = 0:0.5:30;
m0 = [1;0;0];
legend_name = [];
for i=1:1:8
    k32 = k32_list(i);
    legend_name = [legend_name; convertCharsToStrings(sprintf('k32=%.2f',k32_list(i)))];
    
    % ode 45 method
    [t_ode, m_ode] = ode45(@(t, m) model(t, m, k01, k21, k12, k32), tspan, m0);
    m_list_ode(i,:,:) = m_ode;
    
    % exponential matrix method
    K = [-(k01+k21), k21, 0; k21, -(k12+k32), 0; 0, k32, 0];
    cnt = 1;
    for t=0:0.5:30
        m_exp(cnt,:) = expm(K*t)*m0;
        cnt = cnt+1;
    end
    m_list_exp(i,:,:) = m_exp;
end

figure;
for i=1:1:8
    plot(tspan, m_list_ode(i,:,3));
    hold on
end
legend(legend_name)
title('m_3 using ode45')
saveas(gcf, 'p2_ode45.png');

figure;
for i=1:1:8
    plot(tspan, m_list_exp(i,:,3));
    hold on
end
legend(legend_name)
title('m_3 using exponential matrix')
saveas(gcf, 'p2_expmat.png');


figure;
for i=1:1:8
    plot(tspan, m_list_ode(i,:,3) - m_list_exp(i,:,3));
    hold on
end
legend(legend_name)
title('m_3 difference using ode45 and exponential matrix')
saveas(gcf, 'p2_diff.png');


sprintf('largest absolute difference is:%f',max(max(abs(m_list_ode(:,:,3) - m_list_exp(:,:,3)))))

function dmdt = model(t, m, k01, k21, k12, k32)
    dmdt = [-(k01+k21)*m(1)+k12*m(2); k21*m(1)-(k12+k32)*m(2); k32*m(2)];
end