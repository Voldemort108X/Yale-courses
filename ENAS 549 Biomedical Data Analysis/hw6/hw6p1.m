%% problem 1
data = [2.9480, 14.7911, 10.5538, 27.5801, 18.0605, 26.4911, 9.6236, 18.8385, 15.5617, 16.0945, 23.1242, 26.9633, 32.4666]';
n = size(data, 1);
mean_data = mean(data);
std_data = std(data);
ci99_pred = [mean_data-3.0545*std_data/sqrt(n), mean_data+3.0545*std_data/sqrt(n)];

pd = fitdist(data, 'Normal');
ci99 = paramci(pd,'Alpha',.01);

%%
sample = [2.9480, 14.7911, 10.5538, 27.5801, 18.0605, 26.4911, 9.6236, ...
    18.8385, 15.5617, 16.0945, 23.1242, 26.9633, 32.4666];
n = length(sample);
samplemean = mean(sample);
lowerboundary = samplemean - std(sample)*icdf('t',.995,n-1)/sqrt(n);
upperboundary = samplemean + std(sample)*icdf('t',.995,n-1)/sqrt(n);