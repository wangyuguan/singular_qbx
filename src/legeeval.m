function V = legeeval(x, p)
x = x(:);
n = numel(x);
V = zeros(n, p+1);
V(:,1) = 1;
if p == 0, return; end
V(:,2) = x;
for k = 1:p-1
    V(:,k+2) = ((2*k+1).*x.*V(:,k+1) - k.*V(:,k)) / (k+1);
end
end
