%S = load('Reuters21578.mat');
%X = array2table(S.fea);

%y = S.gnd;

%training = S.trainIdx;
%test = S.testIdx;


%X_training = X(training, :); %trainIdx is a vector containing the indices corresponding to the rows in X which comprise the training data
%X_test = X(test, :); %testIdx is a vector containing the indices corresponding to the rows in X which comprise the test data

%y_training = y(S.trainIdx, 1); 
%y_test = y(S.testIdx, 1);

S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

approx_error_nnsc = zeros(8, 1);
error_nnsc = zeros(8, 1);

lambda_v = zeros(8, 1);

epsilon_1 = 10 ^ -8;
epsilon_2 = 10 ^ -6;

lambda = 1;
meu = 0.0001;

X_training = [X(1:40, :); X(51:90, :); X(101:140, :)];
y_training = [y(1:40, :); y(51:90, :); y(101:140, :)];

X_test = [X(41:50, :); X(91:100, :); X(141:150, :)];
y_test = [y(41:50, :); y(91:100, :); y(141:150, :)];

r = rank(X);
[rows, cols] = size(X);

for lambda = 0.001 : 0.3: 2.401

A = randn(rows, r);
S = randn(r, cols);

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;


while true
    A_new = A .* (X * S') ./ (A * (S * S') + lambda);
    A_new = (A_new + abs(A_new))/2;
    
    S_new = S .* (A_new' * X) ./ (A_new' * A_new * S + lambda);
    S_new = (S_new + abs(S_new))/2;
    
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

subplot(1,2,1);
title('Lambda vs error without sparsity');
xlabel('Lambda');
ylabel('Error');
plot(lambda_v, approx_error_nnsc);

subplot(1,2,2);
title('Lambda vs error with sparsity');
xlabel('Lambda');
ylabel('Error');
plot(lambda_v, error_nnsc);