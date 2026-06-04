function [Sx, Sy] = slp_grad_inner_hh(sig_in, inner_src, eval_xyz, zk, eps_fmm, M, h)


nev = size(eval_xyz, 2);
nrm = [0; 0; 1];
q = inner_src.wts(:).*sig_in;
Gx = complex(zeros(M, nev));
Gy = complex(zeros(M, nev));
for m = 1:M
    P = eval_xyz + m*h*nrm;
    g = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, struct('r', P));
    sx = complex(zeros(nev, 1));
    sy = complex(zeros(nev, 1));
    for a = 1:nev
        [~, sxa, sya] = slp_grad(P(:, a), inner_src.r, q, zk);
        sx(a) = sum(sxa);
        sy(a) = sum(sya);
    end
    Gx(m, :) = (sx + g.spmat_x*sig_in).';
    Gy(m, :) = (sy + g.spmat_y*sig_in).';
end

sv = (1:M).';
V = sv.^(0:M-1);
cx = V\Gx;
cy = V\Gy;
Sx = cx(1, :).';
Sy = cy(1, :).';
end
