
function [Sx, Sy] = slp_grad_inner_hh(sig_in, inner_src, eval_xyz, zk, eps_fmm, M, h)
nev = size(eval_xyz, 2);
nf = size(sig_in, 2);
nrm = [0; 0; 1];
src = inner_src.r;
w = inner_src.wts(:);
Gx = complex(zeros(M, nev, nf));
Gy = complex(zeros(M, nev, nf));
for m = 1:M
    P = eval_xyz + m*h*nrm;
    g = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, struct('r', P));
    [~, sx, sy] = eval_fmm(P, src, (w.*sig_in).', zk);
    Gx(m, :, :) = sx + g.spmat_x*sig_in;
    Gy(m, :, :) = sy + g.spmat_y*sig_in;
end
sv = (1:M).';
V = sv.^(0:M-1);
Sx = complex(zeros(nev, nf));
Sy = complex(zeros(nev, nf));
for f = 1:nf
    cx = V\Gx(:, :, f);
    cy = V\Gy(:, :, f);
    Sx(:, f) = cx(1, :).';
    Sy(:, f) = cy(1, :).';
end
end