function [S, Sx, Sy] = smooth_layer_fmm(tgt, src, q, zk)

nd = size(q, 1);
srcinfo = [];
srcinfo.sources = src;
srcinfo.nd = nd;
srcinfo.charges = q;
U = hfmm3d(1e-12, zk, srcinfo, 0, tgt, 2);

if nd == 1
    S = U.pottarg(:);
    Sx = U.gradtarg(1, :).';
    Sy = U.gradtarg(2, :).';
else
    S = U.pottarg.';
    Sx = squeeze(U.gradtarg(:, 1, :)).';
    Sy = squeeze(U.gradtarg(:, 2, :)).';
end
end
