% A computer virus can damage a file with probability 0.2, independently
% of other files. A computer manager checks the condition of important
% files. Conduct a Monte Carlo study to estimate:
% 
% a) the probability that the manager has to check 8 files in order to
%    find 3 damaged ones;
% b) the expected number of clean files found before finding the third
%    damaged one.
% 
% Compare your results with the exact values.

format longg;
p = 0.2;

function N=computeSimulationsCount()
    err = input('max error = '); % maximum error
    alpha = input('alpha (level of significance) = '); % significance level

    % Compute MC size to ensure that the error is < err, with confidence level 1 - alpha
    N = ceil(0.25 * (norminv(alpha / 2, 0, 1) / err)^2); 
    fprintf('Nr. of simulations N = %d\n', N);
end
N = computeSimulationsCount();

% a) The manager has to check 8 files in order to find 3 damaged one.
% Events:
% X - clean files files checked until finding the third damaged one
% - X failures until the nth (third) success, probability of success p
% -> Negative Binomial Distribution
% - success -> damaged file -> probability p
% We need to find P(X == 5)

n = 3;
X = zeros(1, N);
for i = 1:N
    Y = ceil(log(1 - rand(n, 1))/log(1 - p) - 1); % Geo variables
    X(i) = sum(Y);
end

fprintf('simulated probab. P(X = 5) = %g\n', mean(X==5));
fprintf('true probab. P(X = 5) = %g\n', nbinpdf(5, n, p));
fprintf('error= %e\n', abs(nbinpdf(5, n, p)) - mean(X==5));

% b) The number of clean files found before finding the third damaged one
% Events:
% X - clean files found before the nth damaged file
% - X failures before nth success
% -> Negative Binomial Distribution
% - success -> damaged file -> probability p
% We need to find E(X)

fprintf('simulated probab. E(X) = %g\n', mean(X));
fprintf('true probab. E(X) = %g\n', (n * (1 - p) / p));
fprintf('error= %e\n', abs(n * (1 - p) / p - mean(X)));
