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
opts.P = 20;
opts.k_dya = k;
opts.ncores = 16;

D_rho = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, ...
    t_splits, -1/2);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma,...
    t_splits, 1/2);
nb = size(D_rho.src_xyz, 2);

norder = k;
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

dr = (1 - lam_inner)/nch1;
idx = find(lam_in >= lam_inner - 3*dr);


% source on the D_J nodes (alpha = +1/2)
QbiJ = precompute_helm_qbx_corr(inner_src.r(:, idx), lam_in(idx), t_in(idx), D_J, opts, zk);
Qbb = precompute_helm_qbx_corr(edge_tar, lam_ed, t_ed, D_J, opts, zk);
b2i_S_J = sparse(ni, nb);   b2i_S_J(idx, :) = QbiJ.S;
b2i_gx_J = sparse(ni, nb);  b2i_gx_J(idx, :) = QbiJ.Sx;
b2i_gy_J = sparse(ni, nb);  b2i_gy_J(idx, :) = QbiJ.Sy;
b2b_S_J = Qbb.S;  b2b_gx_J = Qbb.Sx;  b2b_gy_J = Qbb.Sy;

% source on the D_rho nodes (alpha = -1/2)
QbiR = precompute_helm_qbx_corr(inner_src.r(:, idx), lam_in(idx), t_in(idx), D_rho, opts, zk);
Qbb = precompute_helm_qbx_corr(edge_tar, lam_ed, t_ed, D_rho, opts, zk);
b2i_S_rho = sparse(ni, nb);   b2i_S_rho(idx, :) = QbiR.S;
b2i_gx_rho = sparse(ni, nb);  b2i_gx_rho(idx, :) = QbiR.Sx;
b2i_gy_rho = sparse(ni, nb);  b2i_gy_rho(idx, :) = QbiR.Sy;
b2b_S_rho = Qbb.S;  b2b_gx_rho = Qbb.Sx;  b2b_gy_rho = Qbb.Sy;

% assemble the correction matrices
M_S_J = [i2i_S.spmat, b2i_S_J; i2b_S.spmat, b2b_S_J];
M_gx_J = [i2i_grad.spmat_x, b2i_gx_J; i2b_grad.spmat_x, b2b_gx_J];
M_gy_J = [i2i_grad.spmat_y, b2i_gy_J; i2b_grad.spmat_y, b2b_gy_J];

M_S_rho = [i2i_S.spmat, b2i_S_rho; i2b_S.spmat, b2b_S_rho];
M_gx_rho = [i2i_grad.spmat_x, b2i_gx_rho; i2b_grad.spmat_x, b2b_gx_rho];
M_gy_rho = [i2i_grad.spmat_y, b2i_gy_rho; i2b_grad.spmat_y, b2b_gy_rho];

% system-matrix entry handle 
Afun = @(i, j) efie2_sysmat_handle(i, j, target_xyz, ...
    src_xyz_J, src_w_J, M_S_J, M_gx_J, M_gy_J, ...
    src_xyz_rho, src_w_rho, M_S_rho, M_gx_rho, M_gy_rho, zk, N);

rx = [target_xyz(1:2, :), target_xyz(1:2, :), target_xyz(1:2, :);
      zeros(1, N), 10*ones(1, N),  20*ones(1, N)];
cx = [src_xyz_J(1:2, :),  src_xyz_J(1:2, :),  src_xyz_rho(1:2, :);
      zeros(1, N), 10*ones(1, N), 20*ones(1, N)];
occ = 400;
rtol = 1e-10;
fopts = [];
fopts.verb = 1;
fopts.symm = 'n';
fopts.Tmax = 1.5;
tic
F = rskel(Afun, rx, cx, occ, rtol, [], fopts);
t_rskel = toc;
w = whos('F'); 
mem = w.bytes/1e6;
fprintf('rskel time/mem: %10.4e (s) / %6.2f (MB)\n', t_rskel, mem)

mem_dense = (3*N)^2*16/1e6; 
compression = mem_dense/mem;
fprintf('factor of compression is %6d\n', compression)

if 1==0
    % check forward map accuracy 
    f1 = @(p) 2*cos(p(1,:))+3*sin(p(2,:));
    f2 = @(p) p(1,:).^2+p(2,:).^2;
    f3 = @(p) (p(1,:)+p(2,:))/2;
    Xs = [f1(src_xyz_J).'; f2(src_xyz_J).'; f3(src_xyz_rho).'];
    y_rskel = rskel_mv(F, Xs);
    A_true = Afun((1:3*N).', (1:3*N).');
    y_true = A_true*Xs;
    rel_err = norm(y_rskel - y_true)/norm(y_true);
    fprintf('relative error of rskel forward map is %.2d\n', rel_err)
end