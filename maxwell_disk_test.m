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
alpha_J = -1/2;
alpha_rho = -1/2;

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
    t_splits, alpha_rho);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma,...
    t_splits, alpha_J);
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

opts.add_grad = true;

dr = (1 - lam_inner)/nch1;
idx = find(lam_in >= lam_inner - 3*dr);


% source on the D_J nodes 
QbiJ = precompute_helm_qbx_corr(inner_src.r(:, idx), lam_in(idx), t_in(idx), D_J, opts, zk);
Qbb = precompute_helm_qbx_corr(edge_tar, lam_ed, t_ed, D_J, opts, zk);
b2i_S_J = sparse(ni, nb);   b2i_S_J(idx, :) = QbiJ.S;
b2i_gx_J = sparse(ni, nb);  b2i_gx_J(idx, :) = QbiJ.Sx;
b2i_gy_J = sparse(ni, nb);  b2i_gy_J(idx, :) = QbiJ.Sy;
b2b_S_J = Qbb.S;  b2b_gx_J = Qbb.Sx;  b2b_gy_J = Qbb.Sy;

% source on the D_rho nodes  
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


rx = [target_xyz, target_xyz, target_xyz];      
cx = [src_xyz_J,  src_xyz_J,  src_xyz_rho];

occ = 500;
rtol = 1e-8;
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


[Asp, pp, qq] = rskel_xsp(F);
[L, U, P] = lu(Asp);
rhs_ext = [rhs(pp); zeros(size(Asp, 1) - N*3, 1)];
sol_ext = U\(L\(P*rhs_ext));
xsol = sol_ext(1:N*3);
xsol(qq) = xsol;


Jx = xsol(1:N);
Jy = xsol(N+1:2*N);
rho = xsol(2*N+1:3*N);





save('maxwell_results.mat', 'Jx', 'Jy', 'rho')

% check boundary condition
M = 5;
h = 1e-2;

ng = 100;
g = linspace(-1, 1, ng);
[X, Y] = meshgrid(g, g);
lam_g = sqrt(X.^2 + Y.^2);
sel = lam_g <= 1;
eval_xyz = [X(sel).'; Y(sel).'; zeros(1, nnz(sel))];
nt = size(eval_xyz, 2);

lam_e = sqrt(eval_xyz(1, :).^2 + eval_xyz(2, :).^2);
t_e = mod(atan2(eval_xyz(2, :), eval_xyz(1, :)), 2*pi).';

targinfo_eval = struct('r', eval_xyz);
i2e_S = helm3d.dirichlet.get_quad_cor_sub(inner_src, eps_fmm, zk, [1, 0], targinfo_eval);
i2e_grad = helm3d.sgrad.get_quad_cor_sub(inner_src, eps_fmm, zk, targinfo_eval);

idx_e = find(lam_e >= lam_inner - 3*dr);
QJ = precompute_helm_qbx_corr(eval_xyz(:, idx_e), lam_e(idx_e), t_e(idx_e), D_J, opts, zk);
QeJ.S = sparse(nt, nb);   
QeJ.S(idx_e, :) = QJ.S;
QeJ.Sx = sparse(nt, nb);  
QeJ.Sx(idx_e, :) = QJ.Sx;
QeJ.Sy = sparse(nt, nb);  
QeJ.Sy(idx_e, :) = QJ.Sy;
QR = precompute_helm_qbx_corr(eval_xyz(:, idx_e), lam_e(idx_e), t_e(idx_e), D_rho, opts, zk);
QeR.S = sparse(nt, nb);   
QeR.S(idx_e, :) = QR.S;
QeR.Sx = sparse(nt, nb);  
QeR.Sx(idx_e, :) = QR.Sx;
QeR.Sy = sparse(nt, nb);  
QeR.Sy(idx_e, :) = QR.Sy;
 

[S_Jx, ~, ~] = eval_layer(Jx, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[S_Jy, ~, ~] = eval_layer(Jy, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[S_rho, ~, ~] = eval_layer(rho, inner_src, D_rho, eval_xyz, zk, i2e_S, i2e_grad, QeR);
Jx_b = [zeros(ni, 1); Jx(ni+1:end)];
Jy_b = [zeros(ni, 1); Jy(ni+1:end)];
rho_b = [zeros(ni, 1); rho(ni+1:end)];
[~, dxJx_b, ~] = eval_layer(Jx_b, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[~, ~, dyJy_b] = eval_layer(Jy_b, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[~, dxrho_b, dyrho_b] = eval_layer(rho_b, inner_src, D_rho, eval_xyz, zk, i2e_S, i2e_grad, QeR);

 
[Sx_in, Sy_in] = slp_grad_inner_hh([Jx(1:ni), Jy(1:ni), rho(1:ni)], inner_src, eval_xyz, zk, eps_fmm, M, h);
dxJx = Sx_in(:, 1) + dxJx_b;
dyJy = Sy_in(:, 2) + dyJy_b;
dx_rho = Sx_in(:, 3) + dxrho_b;
dy_rho = Sy_in(:, 3) + dyrho_b;

Ex = 1i*zk*S_Jx + dx_rho;
Ey = 1i*zk*S_Jy + dy_rho;
divres = 1i*zk*(dxJx + dyJy) - zk^2*S_rho;

E_inc_e = dipole_field(eval_xyz, x0, p0, zk);
Enorm = max(abs(E_inc_e(:)));
errx = abs(Ex + E_inc_e(1, :).')/Enorm;
erry = abs(Ey + E_inc_e(2, :).')/Enorm;
errd = abs(divres)/max(abs(zk^2*S_rho));

errs = {errx, erry, errd};
names_e = {'$E_x$ part', '$E_y$ part', 'div. part'};
tiny = 1e-300;
tcirc = linspace(0, 2*pi, 400);

figure
for c = 1:3
    eg = nan(ng, ng);
    eg(sel) = log10(errs{c} + tiny);
    subplot(1, 3, c)
    him = imagesc(g, g, eg);
    set(him, 'AlphaData', ~isnan(eg));
    axis equal tight
    axis xy
    colorbar
    hold on
    plot(cos(tcirc), sin(tcirc), 'k-', 'LineWidth', 1.5)
    plot(lam_inner*cos(tcirc), lam_inner*sin(tcirc), 'k--', 'LineWidth', 1.5)
    hold off
    title(names_e{c}, 'Interpreter', 'latex', 'FontSize', 16)
    xlabel('$x$', Interpreter='latex', FontSize=16)
    ylabel('$y$', Interpreter='latex', FontSize=16)
end
set(gcf, 'Position', [100, 100, 1400, 420])
exportgraphics(gcf, 'maxwell_PEC_error.pdf', 'ContentType', 'vector');


% check resolution of the solved densities
dens = {Jx, Jy, rho};
Dband = {D_J, D_J, D_rho};
names = {'$J_1$', '$J_2$', '$\rho$'};
tiny = 1e-300;
narc = 8;

figure
for d = 1:3
    err_in = abs(inner_src.surf_fun_error(dens{d}(1:ni)));
    err_band = edge_fun_error(Dband{d}, dens{d}(ni+1:end));
    cl = [min(log10([err_in(:); err_band(:)] + tiny)), max(log10([err_in(:); err_band(:)] + tiny))];

    subplot(1, 3, d)
    inner_src.plot(log10(err_in(:) + tiny));
    hold on
    for j = 1:nch2
        tt = linspace(t_splits(j), t_splits(j+1), narc);
        for i = 1:nch1
            la = Dband{d}.lam_splits(i);
            lb = Dband{d}.lam_splits(i+1);
            xb = [la*cos(tt), lb*cos(fliplr(tt))];
            yb = [la*sin(tt), lb*sin(fliplr(tt))];
            patch(xb, yb, log10(err_band(i, j) + tiny), 'EdgeColor', 'none')
        end
    end
    hold off
    view(2)
    axis equal tight
    clim(cl)
    colorbar
    title(names{d}, 'Interpreter', 'latex', 'FontSize', 16)
end
set(gcf, 'Position', [100, 100, 1200, 400])
exportgraphics(gcf, 'density_resolution.pdf', 'ContentType', 'vector');


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
    sx = complex(zeros(nev, nf));
    sy = complex(zeros(nev, nf));
    for a = 1:nev
        df = P(:, a) - src;
        r = sqrt(sum(df.^2, 1)).';
        coef = (1i*zk*r - 1).*exp(1i*zk*r)./(4*pi*r.^3);
        sx(a, :) = (coef.*df(1, :).'.*w).'*sig_in;
        sy(a, :) = (coef.*df(2, :).'.*w).'*sig_in;
    end
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