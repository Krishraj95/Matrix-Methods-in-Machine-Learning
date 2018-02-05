S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

approx_error_nnsc = zeros(8, 1);
error_nnsc = zeros(8, 1);

lambda_v = zeros(8, 1);

epsilon_1 = 10 ^ -2;
epsilon_2 = 10 ^ -5;

lambda = 4;
meu = 0.0001;

[rows, cols] = size(X);
r = min(rows, cols);

k = 1;

for lambda = 0.001 : 0.3: 2.401
epsilon1 = 0;
epsilon2 = 0;

A = randn(rows, r);
S = zeros(r, cols);

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

w_new = zeros(1, r);
w = zeros(1, r);

while true 
    for i = 1:rows
        epsilon_w = 1000;
        w_new = A(i, :);
      
        while epsilon_w >= epsilon_1
            w = w_new;
        
            w_new = w - meu * (w * S - X(i, :)) * S';
            epsilon_w = norm(w_new - w);
        end
    
        A(i, :) = (w_new + abs(w_new))/2;
    
    end
    
    S_new = S .* (A' * X) ./ (A' * A * S + lambda);
    S_new = (S_new + abs(S_new))/2;
    
    epsilon2 = norm(S_new - S, 'fro') / sqrt(r * cols);
    
    if (epsilon2 < epsilon_2) %when the new S is not very different from old S
        break; %no point in continuing
    end
    
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
