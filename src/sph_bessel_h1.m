function [jn, hn] = sph_bessel_h1(z, P)
z = z(:).';
M = numel(z);
jn = zeros(P+1, M);
hn = zeros(P+1, M);

sqrtfac = sqrt(pi./(2*z));
for n = 0:P
    Jn = besselj(n+0.5, z);
    Yn = bessely(n+0.5, z);
    jn(n+1, :) = sqrtfac.*Jn;
    hn(n+1, :) = sqrtfac.*(Jn + 1i * Yn);
end
end
