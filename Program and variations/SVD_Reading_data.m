%%
S = load('fisheriris.mat');

X = S.meas .* (S.meas > 0);
y = [ones(50,1); 2 .* ones(50,1); 3 .* ones(50,1)];

epsilon = 10 ^ -1;
lambda = 1;
meu = 0.1;

[rows, cols] = size(X);

[U, Sigma, V] = svd(X, 'econ');
r = nnz(Sigma);

sum = 0;
S = Sigma * V';

for i = 1:cols
    sum = sum + norm(S(:, i), 1);
end 

approx_error_svd = norm(X - U * S,'fro') / sqrt(rows * cols);

error_svd = approx_error_svd + lambda * sum;

[W, H, D] = nnmf(X, r);
sum = 0;

for i = 1:cols
    sum = sum + norm(H(:, i), 1);
end 

approx_error_nnmf = norm(X - W * H,'fro') / sqrt(rows * cols);

error_nnmf = approx_error_nnmf + lambda * sum;

%%
X = U * Sigma * V';

X_training = [X(1:40, :); X(51:90, :); X(101:140, :)];
y_training = [y(1:40, :); y(51:90, :); y(101:140, :)];

X_test = [X(41:50, :); X(91:100, :); X(141:150, :)];
y_test = [y(41:50, :); y(91:100, :); y(141:150, :)];

Mdl = fitcecoc(X_training,y_training);

label = predict(Mdl, X_test);
error_classifier_svd = nnz(y_test - label);

%%
X = W * H;

X_training = [X(1:40, :); X(51:90, :); X(101:140, :)];
y_training = [y(1:40, :); y(51:90, :); y(101:140, :)];

X_test = [X(41:50, :); X(91:100, :); X(141:150, :)];
y_test = [y(41:50, :); y(91:100, :); y(141:150, :)];

Mdl = fitcecoc(X_training,y_training);

label = predict(Mdl, X_test);
error_classifier_nnmf = nnz(y_test - label);