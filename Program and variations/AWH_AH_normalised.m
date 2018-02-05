S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

epsilon_1 = 10 ^ -8;
epsilon_2 = 10 ^ -5;

lambda = 1;
meu = 0.0001;

r = rank(X);
[rows, cols] = size(X);

A = randn(rows, r);
S = randn(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

while true
    A_new = (A - meu * (A * S - X) * S');
    A_new = (A_new + abs(A_new))/2;
     
    for i = 1:r
        A_new(:, i) = A_new(:, i)/norm(A_new(:, i)); %we normalise the ith column of A_new and put the new column into A at the same position as i
    end
    
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

S = S_new;
X_S = S';
%S = W * H;

[row_S, col_S] = size(X_S);
r_S = rank(X_S);

W_S = randn(row_S, r_S);
H_S = randn(r_S, col_S);

W_S = (W_S + abs(W_S))/2;
H_S = (H_S + abs(H_S))/2;

epsilon1_S = 1000;
epsilon2_S = 1000;

for i = 1:r_S
   W_S(:, i) = W_S(:, i)/norm(W_S(:, i)); %we normalise the ith column of H
end

while true
    W_S_new = (W_S - meu * (W_S * H_S - X_S) * H_S');
    W_S_new = (W_S_new + abs(W_S_new))/2;
    
    for i = 1:r_S
        W_S_new(:, i) = W_S_new(:, i)/norm(W_S_new(:, i)); %we normalise the ith column of H
    end
    
    H_S_new = H_S .* (W_S_new' * X_S) ./ (W_S_new' * W_S_new * H_S + lambda);
    H_S_new = (H_S_new + abs(H_S_new))/2;
    
    epsilon1_S = norm(W_S_new - W_S, 'fro') / sqrt(row_S * r_S);
    epsilon2_S = norm(H_S_new - H_S, 'fro') / sqrt(r_S * col_S);
    
    if (epsilon1_S < epsilon_1 && epsilon2_S < epsilon_2) %when the new A is not very different from old A
        break; %no point in continuing
    end
    
    W_S = W_S_new;
    H_S = H_S_new;  
end

sum = 0;

for i = 1:col_S
    sum = sum + norm(H_S_new(:, i), 1);
end 

approx_error_nnsc = norm(X - A_new * (W_S_new * H_S_new)', 'fro') / sqrt(rows * cols);

error_nnsc = approx_error_nnsc + lambda * sum;
