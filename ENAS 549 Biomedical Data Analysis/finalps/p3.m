% logan_sim.m  
% simuate Logan graphical analysis
close all; clear all;

% input function
t_cp=[0:.1:10,11:60,65:5:120];
cp=exponentials([100,.1,20,.01],t_cp);

% rate constants
% k1=.4; k2=.2; k3=.05; k4=.02;
k1=.3; k2=.15; k3=.05; k4=.01;
k=[k1,k2,k3,k4];
%total volume of distribution
vt=k1/k2*(1+k3/k4);
%y0=[0,0];
%extra.k=k;
% Save input curve in structure for pet_model
extra.cp=cp;
extra.t_cp=t_cp;
% simulate perfect tissue data
dur_ct=horzcat(ones(1,6),2,2,ones(1,22)*5); % scan durations
t_ct=cumsum(dur_ct)-dur_ct/2.;  % mid time of scan
n=length(t_ct);
%
% seems that if I pass the result times directly, the accuracy is worse
% with the way we sample the input function
%opt=odeset('MaxStep',1.);
%sol=ode45(@pettis,[0,max(t_ct)],y0,opt,extra);
%y=deval(sol,t_ct);
%ct=y(1,:)+y(2,:);
ct = pet_model(k,t_ct,extra);
ct=ct';
figure(1);clf;
plot(t_ct,ct,'k',t_ct,ct,'x');
title('Perfect data'); xlabel('Time (min)'); ylabel('Concentration');
%
% calculate perfect Logan plot
% calculate integral(ct) to times ct_t
intct=cumtrapz(t_ct,ct);
% calculate intergral (cp) to times ct_t - different time sync
intcp=cumtrapz(t_cp,cp);
intcp=interp1(t_cp,intcp,t_ct);
% Logan variables
yl=intct./ct;
xl=intcp./ct;
figure(2);clf;
hold on;
plot(xl,yl,'x');
title('Logan plot'); xlabel('int(Cp)/Ct'); ylabel('int(Ct)/Ct');

% fit points according to the choice of t_star
% t_star_list = 30:5:100;
psd = 1; % assume 1 percent of error
cnt = 1;
nl=length(xl);
for t_star = 30:5:100
    startfit = find(t_ct>t_star, 1, 'first');
    sd=psd/100.*mean(ct);
    nrep=25;
    vall=zeros(1,nrep);
    for i=1:nrep
       ctn=ct+randn(1,n)*sd;
       intctn=cumtrapz(t_ct,ctn);
        % Logan variables
        yl=intctn./ctn;
        xl=intcp./ctn;
        xl2=xl(startfit:nl);
        yl2=yl(startfit:nl);
        b=polyfit(xl2,yl2,1);
        vall(1,i)=b(1);
    end
    display(strcat(['nfit',num2str(startfit), ' t_star=',num2str(t_star),' Percent SD=',num2str(psd),' mean(V)=',num2str(mean(vall)),' std(V)=',num2str(std(vall))]));
    mean_slope_list(cnt) = mean(vall);
    std_slope_list(cnt) = std(vall);
    cnt = cnt + 1;
end
figure;
errorbar(30:5:100,mean_slope_list,std_slope_list);
hold on
yline(vt, '--');
ylim([5,13]);
legend('V_T estimates','true V_T')
xlabel('t*');
saveas(gcf,'p3.png');