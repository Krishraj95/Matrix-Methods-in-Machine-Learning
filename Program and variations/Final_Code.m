%S = load('Reuters21578.mat');
%X = array2table(S.fea);

%y = S.gnd;

%training = S.trainIdx;
%test = S.testIdx;


%X_training = X(training, :); %trainIdx is a vector containing the indices corresponding to the rows in X which comprise the training data
%X_test = X(test, :); %testIdx is a vector containing the indices corresponding to the rows in X which comprise the test data

%y_training = y(S.trainIdx, 1); 
%y_test = y(S.testIdx, 1);
approx_error_nnsc = zeros(7, 1);
error_nnsc = zeros(7, 1);

lambda_v = zeros(7, 1);
S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

epsilon_1 = 10 ^ -6;
epsilon_2 = 10 ^ -6;

%lambda = 1;
meu = 0.0001;

X_training = [X(1:40, :); X(51:90, :); X(101:140, :)];
y_training = [y(1:40, :); y(51:90, :); y(101:140, :)];

X_test = [X(41:50, :); X(91:100, :); X(141:150, :)];
y_test = [y(41:50, :); y(91:100, :); y(141:150, :)];

r = rank(X);
[rows, cols] = size(X);

k = 1;

for lambda = 0.001 : 0.3: 2.101

A = randn(rows, r);
S = zeros(r, cols);

epsilon1 = 0;
epsilon2 = 0;

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

while true
    W = A' * A + lambda * eye(r);
    
    for j = 1:cols
        b = A' * X(:, j);
        
        S_new(:, j) = W \ b;
    end
    S_new = (S_new + abs(S_new))/2;
    
    A_new = A .* (X * S_new') ./ (A * (S_new * S_new') + lambda);
    A_new = (A_new + abs(A_new))/2;
    
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
scatter(lambda_v, approx_error_nnsc, 'o', 'filled');

subplot(1,3,2);
title('Lambda vs sparsity');
xlabel('Lambda');
ylabel('Error');
scatter(lambda_v, error_nnsc, 'o', 'filled');

subplot(1,3,3);
title('Error vs sparsity');
xlabel('Sparsity');
ylabel('Error');
scatter(error_nnsc, approx_error_nnsc, 'o', 'filled');