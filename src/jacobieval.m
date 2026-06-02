function V = jacobieval(x, alpha, beta, p)
x = x(:);
m = numel(x);
V = zeros(m, p+1);
V(:, 1) = 1;
if p >= 1
    V(:, 2) = 0.5*(alpha - beta) + 0.5*(alpha + beta + 2)*x;
end
for n = 2:p
    c = 2*n + alpha + beta;
    a1 = 2*n*(n + alpha + beta)*(c - 2);
    a2 = c - 1;
    a3 = c*(c - 2);
    a4 = alpha^2 - beta^2;
    a5 = 2*(n + alpha - 1)*(n + beta - 1)*c;
    V(:, n+1) = (a2*((a3*x + a4).*V(:, n)) - a5*V(:, n-1))/a1;
end
end
