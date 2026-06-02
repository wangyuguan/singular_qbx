function jn = sph_jn(z, P)

z = z(:).';
M = numel(z);
jn = zeros(P+1, M);
sqrtfac = sqrt(pi ./ (2*z));
for n = 0:P
    jn(n+1, :) = sqrtfac .* besselj(n + 0.5, z);
end
end
