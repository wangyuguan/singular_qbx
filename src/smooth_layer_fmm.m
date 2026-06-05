function [S, Sx, Sy] = smooth_layer_fmm(tgt, src, q, zk)
% smooth single-layer value and (x, y) gradient at targets tgt from sources src
% with charge strengths q (nd x ns, one row per density), via the Helmholtz FMM.
% FMM3D uses the kernel e^{ikr}/(4 pi r), matching our convention, so no rescale.
% Returns S, Sx, Sy as nt x nd.

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
