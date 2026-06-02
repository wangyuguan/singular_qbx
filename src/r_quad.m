function [r, wr] = r_quad(k, a, b, alpha)
if abs(b - 1) < 1e-12
    [r, wr] = jacpts(k, alpha, 0, [a, b]);
    r = r(:);
    wr = wr(:) .* (1 - r).^(-alpha);
else
    [r, wr] = lgwt(k, a, b);
    r = r(:);
    wr = wr(:);
end
end
