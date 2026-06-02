clear 
clc 

run('../fmm3dbie_hirax/matlab/startup.m')
addpath('../FMM3D/matlab/')
run('../FLAM/startup.m')
addpath('../chebfun')
addpath('src/')


rng(1)

zk = 1.0;
eps_fmm = 1e-12;
lam_inner = 0.9;
k = 8;
nch1 = 2;
nch2 = 24;

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
opts.P = 30;
opts.nch_lam_dya = 12;
opts.nch_t_dya = 16;
opts.k_dya = 16;


D_rho = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, ...
    t_splits, -1/2);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma,...
    t_splits, 1/2);
nb = size(D_rho.src_xyz, 2);

norder = 6;
inner_src = geometries.disk([1, 1], [], [4 4 4], norder);
inner_src = scale(inner_src, lam_inner);
ni = inner_src.npts;

N = ni+nb;

target_xyz = [inner_src.r, D_rho.tar_xyz];
src_xyz_J = [inner_src.r, D_J.src_xyz];
src_xyz_rho = [inner_src.r, D_rho.src_xyz];
src_w_J = [inner_src.wts(:); D_J.src_w(:)];
src_w_rho = [inner_src.wts(:); D_rho.src_w(:)];

% right hand side
x0 = [0.0; 0.0; 2.5];
p0 = [1.0; 1.0; 1.0];
E_inc = dipole_field(target_xyz, x0, p0, zk);
rhs = [-E_inc(1, :).'; -E_inc(2, :).'; zeros(N, 1)];

% quadrature correction due to source in the inner part 
i2i_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, ...
    zk, [1, 0]);
i2i_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk);
targinfo_edge = struct('r', D_rho.tar_xyz);
i2b_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, zk,...
    [1, 0], targinfo_edge);
i2b_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk,...
    targinfo_edge);



% quadrature correction due to source in the edge part

lam_in = sqrt(inner_src.r(1, :).^2 + inner_src.r(2, :).^2);
t_in = mod(atan2(inner_src.r(2, :), inner_src.r(1, :)), 2*pi).';

edge_tar = D_rho.tar_xyz;
lam_ed = D_rho.tar_lam_nodes_all;
t_ed = D_rho.tar_t_nodes_all;


% S and grad corrections come out of one pass per band (add_grad = true)
opts.add_grad = true;

% source on the D_J band (alpha = +1/2)
Qbi = precompute_helm_qbx_corr(inner_src.r, lam_in, t_in, D_J, opts, zk);
Qbb = precompute_helm_qbx_corr(edge_tar, lam_ed, t_ed, D_J, opts, zk);
b2i_S_J = Qbi.S;  b2i_gx_J = Qbi.Sx;  b2i_gy_J = Qbi.Sy;
b2b_S_J = Qbb.S;  b2b_gx_J = Qbb.Sx;  b2b_gy_J = Qbb.Sy;

% source on the D_rho band (alpha = -1/2)
Qbi = precompute_helm_qbx_corr(inner_src.r, lam_in, t_in, D_rho, opts, zk);
Qbb = precompute_helm_qbx_corr(edge_tar, lam_ed, t_ed, D_rho, opts, zk);
b2i_S_rho = Qbi.S;  b2i_gx_rho = Qbi.Sx;  b2i_gy_rho = Qbi.Sy;
b2b_S_rho = Qbb.S;  b2b_gx_rho = Qbb.Sx;  b2b_gy_rho = Qbb.Sy;

% assemble the near-correction matrices
M_S_J = [i2i_S.spmat, b2i_S_J; i2b_S.spmat, b2b_S_J];
M_gx_J = [i2i_grad.spmat_x, b2i_gx_J; i2b_grad.spmat_x, b2b_gx_J];
M_gy_J = [i2i_grad.spmat_y, b2i_gy_J; i2b_grad.spmat_y, b2b_gy_J];

M_S_rho = [i2i_S.spmat, b2i_S_rho; i2b_S.spmat, b2b_S_rho];
M_gx_rho = [i2i_grad.spmat_x, b2i_gx_rho; i2b_grad.spmat_x, b2b_gx_rho];
M_gy_rho = [i2i_grad.spmat_y, b2i_gy_rho; i2b_grad.spmat_y, b2b_gy_rho];

% system-matrix entry handle (3N x 3N), x = [J_u; J_v; rho]
Afun = @(i, j) sysmat_handle(i, j, target_xyz, ...
    src_xyz_J, src_w_J, M_S_J, M_gx_J, M_gy_J, ...
    src_xyz_rho, src_w_rho, M_S_rho, M_gx_rho, M_gy_rho, zk, N);

x_pts = [target_xyz(1:2, :), target_xyz(1:2, :), target_xyz(1:2, :);
         0*ones(1, N),       1*ones(1, N),       2*ones(1, N)];
occ = 200;
rtol = 1e-10;
fopts = [];
fopts.verb = 1;
fopts.symm = 'n';
fopts.Tmax = 1.5;
F = rskelf(Afun, x_pts, occ, rtol, [], fopts);
xsol = rskelf_sv(F, rhs);

J_u = xsol(1:N);
J_v = xsol(N+1:2*N);
rho = xsol(2*N+1:3*N);

% accuracy test 

ne = 5;
th_i = 2*pi*rand(1, ne);
r_i = 0.2 + 0.6*rand(1, ne);                            
th_b = 2*pi*rand(1, ne);
r_b = lam_inner + (1 - lam_inner)*(0.1 + 0.8*rand(1, ne)); 
eval_xy = [r_i.*cos(th_i), r_b.*cos(th_b);
           r_i.*sin(th_i), r_b.*sin(th_b)];
eval_xyz = [eval_xy; zeros(1, 2*ne)];


targinfo_eval = struct('r', eval_xyz);
i2e_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, zk, [1, 0], targinfo_eval);
i2e_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, targinfo_eval);

lam_e = sqrt(eval_xyz(1, :).^2 + eval_xyz(2, :).^2);
t_e = mod(atan2(eval_xyz(2, :), eval_xyz(1, :)), 2*pi).';
opts.add_grad = true;
Qe_J = precompute_helm_qbx_corr(eval_xyz, lam_e, t_e, D_J, opts, zk);
Qe_rho = precompute_helm_qbx_corr(eval_xyz, lam_e, t_e, D_rho, opts, zk);

[S_Ju, ~, ~] = eval_layer(J_u, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, Qe_J);
[S_Jv, ~, ~] = eval_layer(J_v, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, Qe_J);
[~, Sx_rho, Sy_rho] = eval_layer(rho, inner_src, D_rho, eval_xyz, zk, i2e_S, i2e_grad, Qe_rho);

Ex = 1i*zk*S_Ju + Sx_rho;
Ey = 1i*zk*S_Jv + Sy_rho;
E_inc_e = dipole_field(eval_xyz, x0, p0, zk);
err_x = Ex + E_inc_e(1, :).';
err_y = Ey + E_inc_e(2, :).';
Enorm = max(abs(E_inc_e(:)));
err = sqrt(abs(err_x).^2 + abs(err_y).^2)/Enorm;

err 
