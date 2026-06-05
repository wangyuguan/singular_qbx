function [S, Sx, Sy] = eval_layer(sigma, inner_src, D, eval_xyz, zk, i2e_S, i2e_grad, Qe)

ni = inner_src.npts;
sig_in = sigma(1:ni);
sig_bl = sigma(ni+1:end);

% inner part
[Si, Sxi, Syi] = smooth_layer_fmm(eval_xyz, inner_src.r, (inner_src.wts(:).*sig_in).', zk);
Si = Si + i2e_S.spmat*sig_in;
Sxi = Sxi + i2e_grad.spmat_x*sig_in;
Syi = Syi + i2e_grad.spmat_y*sig_in;

% edge part
[Sb, Sxb, Syb] = smooth_layer_fmm(eval_xyz, D.src_xyz, (D.src_w(:).*sig_bl).', zk);
Sb = Sb + Qe.S*sig_bl;
Sxb = Sxb + Qe.Sx*sig_bl;
Syb = Syb + Qe.Sy*sig_bl;

S = Si + Sb;
Sx = Sxi + Sxb;
Sy = Syi + Syb;
end

