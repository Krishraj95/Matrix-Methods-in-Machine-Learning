S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

epsilon_1 = 10 ^ -8;
epsilon_2 = 10 ^ -6;

lambda = 0.1;
meu = 0.0001;


X_training = [X(1:40, :); X(51:90, :); X(101:140, :)];
y_training = [y(1:40, :); y(51:90, :); y(101:140, :)];

X_test = [X(41:50, :); X(91:100, :); X(141:150, :)];
y_test = [y(41:50, :); y(91:100, :); y(141:150, :)];

X_S = X';

r = rank(X_S);
[rows, cols] = size(X_S);

A = randn(rows, r);
S = randn(r, cols);

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

while true
    A_new = (A - meu * (A * S - X_S) * S');
    A_new = (A_new + abs(A_new))/2;
    
    for i = 1:r
        A_new(:, i) = A_new(:, i)/norm(A_new(:, i)); %we normalise the ith column of A_new and put the new column into A at the same position as i
    end
    
    S_new = S .* (A_new' * X_S) ./ (A_new' * A_new * S + lambda);
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

approx_error_nnsc = norm(X - (A_new * S_new)', 'fro') / sqrt(rows * cols);

error_nnsc = approx_error_nnsc + lambda * sum;
