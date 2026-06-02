function jnp = sph_jn_deriv(z, P, jn)
z = z(:).';
M = numel(z);
jnp = zeros(P+1, M);
jnp(1, :) = -jn(2, :);
for n = 1:P
    jnp(n+1, :) = jn(n, :) - ((n+1) ./ z) .* jn(n+1, :);
end
end
