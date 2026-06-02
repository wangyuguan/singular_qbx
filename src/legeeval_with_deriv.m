function [V, Vp] = legeeval_with_deriv(x, p)
x = x(:);
n = numel(x);
V  = zeros(n, p+1);
Vp = zeros(n, p+1);
V(:,1) = 1;
Vp(:,1) = 0;
if p == 0
    return; 
end
V(:,2) = x;
Vp(:,2) = 1;
for k = 1:p-1
    V(:, k+2)  = ((2*k+1)*x.*V(:,k+1) - k*V(:,k)) / (k+1);
    Vp(:, k+2) = (2*k+1)*V(:,k+1) + Vp(:,k);
end
end
