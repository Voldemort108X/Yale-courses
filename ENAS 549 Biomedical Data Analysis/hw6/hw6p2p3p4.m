%% problem 3
p1 = [97, 109, 82, 94, 90, 81, 122, 86, 69, 117, 95];
p2 = [63, 89, 99, 67, 82, 78, 63, 98, 80, 88, 69];
[h, p, ci, stats] = ttest(p1, p2, 'Tail', 'right');

% T = sqrt(11)*(mean(p1)-mean(p2))./(std(p1-p2));
% p_manual = cdf('t', T, 10);

%% problem 4
n_rep = 1000;
mu1 = 65;
mu2 = 50;
n = 10;

cnt_sigma = 1;

for sigma = 10:10:50
    cnt_success_eqvar = 0;
    cnt_success_uneqvar = 0;
    
    for i = 1:1:n_rep
        g1 = mu1 + sigma * randn(n, 1);
        g2 = mu2 + 10 * randn(n, 1);

        [h_eqvar, p_eqvar, ci_eqvar, stats_eqvar] = ttest2(g1, g2);
        [h_uneqvar, p_uneqvar, ci_uneqvar, stats_uneqvar] = ttest2(g1, g2, 'Vartype', 'unequal'); 

        if p_eqvar<0.05
            cnt_success_eqvar = cnt_success_eqvar + 1;
        end
        if p_uneqvar<0.05
            cnt_success_uneqvar = cnt_success_uneqvar + 1;
        end
    end
    cnt_suc_eqvar(cnt_sigma) = cnt_success_eqvar;
    cnt_suc_uneqvar(cnt_sigma) = cnt_success_uneqvar;
    
    cnt_sigma = cnt_sigma + 1;
end