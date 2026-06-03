function [Sx, Sy] = eval_layer_grad_hh(sigma, inner_src, D, eval_xyz, zk, eps_fmm, opts, M, h, lam_gate)

nb = size(D.src_xyz, 2);
nev = size(eval_xyz, 2);
nrm = [0; 0; 1];
lam_e = sqrt(eval_xyz(1, :).^2 + eval_xyz(2, :).^2);
t_e = mod(atan2(eval_xyz(2, :), eval_xyz(1, :)), 2*pi).';
idx_e = find(lam_e >= lam_gate);

Gx = complex(zeros(M, nev));
Gy = complex(zeros(M, nev));
for m = 1:M
    P = eval_xyz + m*h*nrm;
    i2e_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, zk, [1, 0], struct('r', P));
    i2e_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, struct('r', P));
    Q = precompute_helm_qbx_corr(P(:, idx_e), lam_e(idx_e), t_e(idx_e), D, opts, zk);
    Qe.S = sparse(nev, nb);   Qe.S(idx_e, :) = Q.S;
    Qe.Sx = sparse(nev, nb);  Qe.Sx(idx_e, :) = Q.Sx;
    Qe.Sy = sparse(nev, nb);  Qe.Sy(idx_e, :) = Q.Sy;
    [~, Sx_m, Sy_m] = eval_layer(sigma, inner_src, D, P, zk, i2e_S, i2e_grad, Qe);
    Gx(m, :) = Sx_m.';
    Gy(m, :) = Sy_m.';
end

s = (1:M).';
V = s.^(0:M-1);
cx = V\Gx;
cy = V\Gy;
Sx = cx(1, :).';
Sy = cy(1, :).';
end
