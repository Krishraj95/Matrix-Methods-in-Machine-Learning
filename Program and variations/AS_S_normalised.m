S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

approx_error_nnsc = zeros(6, 1);
error_nnsc = zeros(6, 1);

lambda_v = zeros(6, 1);

epsilon_1 = 10 ^ -4;
epsilon_2 = 10 ^ -4;

lambda = 0;
meu = 0.0001;

r = rank(X);
[rows, cols] = size(X);

k = 1;

for lambda = 0.001 : 0.3: 1.801
epsilon1 = 0;
epsilon2 = 0;
    
A = randn(rows, r);
S = zeros(r, cols);

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:cols
   S(:, i) = S(:, i)/norm(S(:, i)); %we normalise the ith column of S
end

while true
    A_new = (A - meu * (A * S - X) * S');
    A_new = (A_new + abs(A_new))/2;
    
    S_new = S .* (A_new' * X) ./ (A_new' * A_new * S + lambda);
    S_new = (S_new + abs(S_new))/2;
    
    for i = 1:cols
        S_new(:, i) = S_new(:, i)/norm(S_new(:, i)); %we normalise the ith column of S_new and put the new column into S_new at the same position as i
    end
    
    epsilon1 = norm(A_new - A, 'fro') / sqrt(rows * r);
    epsilon2 = norm(S_new - S, 'fro') / sqrt(r * cols);
    
    if (epsilon1 < epsilon_1 && epsilon2 < epsilon_2) %when the new A is not very different from old A
        break; %no point in continuing
    end
    
    A = A_new;
    S = S_new;  
end

sum = 0;

for i = 1:cols
    sum = sum + norm(S_new(:, i), 1);
end 

approx_error_nnsc(k, 1) = norm(X - A_new * S_new, 'fro') / sqrt(rows * cols);

error_nnsc(k, 1) = sum;
lambda_v(k, 1) = lambda;

k = k + 1;
end

figure;

subplot(1,3,1);
title('Lambda vs error');
xlabel('Lambda');
ylabel('Error');
plot(lambda_v, approx_error_nnsc);

subplot(1,3,2);
title('Lambda vs sparsity');
xlabel('Lambda');
ylabel('Error');
plot(lambda_v, error_nnsc);

subplot(1,3,3);
title('Error vs sparsity');
xlabel('Sparsity');
ylabel('Error');
plot(error_nnsc, approx_error_nnsc);