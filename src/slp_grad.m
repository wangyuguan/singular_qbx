function [S, Sx, Sy, Sz] = slp_grad(tgt, src, w, zk)
% Helmholtz single-layer value S and its gradient (w.r.t. tgt) between points
% tgt and src, weighted by w.  df = tgt - src broadcasts, so one of tgt/src may
% be a single point and the other many.  Self interactions (r ~ 0) are set to 0.
%   S  = w  e^{i k r} / (4 pi r)
%   Sx = w (i k r - 1) e^{i k r} / (4 pi r^3) (tgt - src)_x   (and y, z)
df = tgt - src;
r = sqrt(sum(df.^2, 1)).';
self = r < 1e-14;
eikr = exp(1i*zk*r);
S = eikr./(4*pi*r);
coef = (1i*zk*r - 1).*eikr./(4*pi*r.^3);
S(self) = 0;
coef(self) = 0;
w = w(:);
S = w.*S;
Sx = w.*coef.*df(1, :).';
Sy = w.*coef.*df(2, :).';
Sz = w.*coef.*df(3, :).';
end
