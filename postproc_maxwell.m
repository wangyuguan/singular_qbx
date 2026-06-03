 
 
clear
clc

run('../fmm3dbie_hirax/matlab/startup.m')
addpath('../FMM3D/matlab/')
addpath('../chebfun')
addpath('src/')

 
load('maxwell_results.mat', 'Jx', 'Jy', 'rho')

 
zk = 1.0;
eps_fmm = 1e-12;
lam_inner = 0.9;
k = 8;
nch1 = 2;
nch2 = 24;
M = 5;
h = 1e-2;

gamma = @(t) [cos(t); sin(t); 0];
dgamma = @(t) [-sin(t); cos(t); 0];
t_splits = linspace(0, 2*pi, nch2+1);

opts = [];
opts.n_lam_near = 1;
opts.n_t_near = 1;
opts.nsub = 4;
opts.k_sub = 16;
opts.n_lam_near_sub = 1;
opts.n_t_near_sub = 1;
opts.P = 20;
opts.k_dya = k;
opts.ncores = 16;
opts.add_grad = true;

D_rho = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, -1/2);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, 1/2);
nb = size(D_rho.src_xyz, 2);

norder = k;
inner_src = geometries.disk([1, 1], [], [4 4 4], norder);
inner_src = scale(inner_src, lam_inner);
ni = inner_src.npts;

x0 = [0.0; 0.0; 2.5];
p0 = [1.0; 1.0; 1.0];

 
ng = 100;
g = linspace(-1, 1, ng);
[X, Y] = meshgrid(g, g);
lam_g = sqrt(X.^2 + Y.^2);
sel = lam_g <= 1;
eval_xyz = [X(sel).'; Y(sel).'; zeros(1, nnz(sel))];
nev = size(eval_xyz, 2);

lam_e = sqrt(eval_xyz(1, :).^2 + eval_xyz(2, :).^2);
t_e = mod(atan2(eval_xyz(2, :), eval_xyz(1, :)), 2*pi).';

 
targinfo_eval = struct('r', eval_xyz);
i2e_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, zk, [1, 0], targinfo_eval);
i2e_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, targinfo_eval);

 
dr = (1 - lam_inner)/nch1;
idx_e = find(lam_e >= lam_inner - 3*dr);
QJ = precompute_helm_qbx_corr(eval_xyz(:, idx_e), lam_e(idx_e), t_e(idx_e), D_J, opts, zk);
QeJ.S = sparse(nev, nb);   QeJ.S(idx_e, :) = QJ.S;
QeJ.Sx = sparse(nev, nb);  QeJ.Sx(idx_e, :) = QJ.Sx;
QeJ.Sy = sparse(nev, nb);  QeJ.Sy(idx_e, :) = QJ.Sy;
QR = precompute_helm_qbx_corr(eval_xyz(:, idx_e), lam_e(idx_e), t_e(idx_e), D_rho, opts, zk);
QeR.S = sparse(nev, nb);   QeR.S(idx_e, :) = QR.S;
QeR.Sx = sparse(nev, nb);  QeR.Sx(idx_e, :) = QR.Sx;
QeR.Sy = sparse(nev, nb);  QeR.Sy(idx_e, :) = QR.Sy;

 
[S_Jx, ~, ~] = eval_layer(Jx, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[S_Jy, ~, ~] = eval_layer(Jy, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);

 
rho_band = [zeros(ni, 1); rho(ni+1:end)];
[~, Sxb, Syb] = eval_layer(rho_band, inner_src, D_rho, eval_xyz, zk, i2e_S, i2e_grad, QeR);
[Sxi, Syi] = slp_grad_inner_hh(rho(1:ni), inner_src, eval_xyz, zk, eps_fmm, M, h);
Sx_rho = Sxi + Sxb;
Sy_rho = Syi + Syb;

Ex = 1i*zk*S_Jx + Sx_rho;
Ey = 1i*zk*S_Jy + Sy_rho;

 
E_inc_e = dipole_field(eval_xyz, x0, p0, zk);
Enorm = max(abs(E_inc_e(:)));
err = sqrt(abs(Ex + E_inc_e(1, :).').^2 + abs(Ey + E_inc_e(2, :).').^2)/Enorm;

 

figure
scatter(eval_xyz(1, :), eval_xyz(2, :), 25, log10(err), 'filled')
axis equal
colorbar
xlabel('$x$', Interpreter='latex', FontSize=16)
ylabel('$y$', Interpreter='latex', FontSize=16)
exportgraphics(gcf, 'maxwell_PEC_error.pdf', 'ContentType', 'vector');

