function hn = sph_h1(z, P)
z = z(:).';
M = numel(z);
hn = complex(zeros(P+1, M));
e = exp(1i*z);
hn(1, :) = -1i*e./z;
if P >= 1
    hn(2, :) = -e.*(1./z + 1i./z.^2);
end
for n = 1:P-1
    hn(n+2, :) = (2*n+1)./z.*hn(n+1, :) - hn(n, :);
end
end
