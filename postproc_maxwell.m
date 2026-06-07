
clear
clc

run('../fmm3dbie_hirax/matlab/startup.m')
addpath('../FMM3D/matlab/')
addpath('../chebfun')
addpath('src/')

load('maxwell_results.mat', 'Jx', 'Jy', 'rho')


zk = 1.0;
eps_fmm = 1e-12;
lam_inner = 0.95;
k = 8;
nch1 = 1;
nch2 = 36;
alpha_J = -1/2;
alpha_rho = -1/2;
tar_type = 'gl';

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

D_rho = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha_rho, tar_type);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, alpha_J, tar_type);
nb = size(D_rho.src_xyz, 2);

norder = k;
inner_src = geometries.disk([1, 1], [], [4 4 4], norder);
inner_src = scale(inner_src, lam_inner);
ni = inner_src.npts;

N = ni + nb;
if numel(Jx) ~= N
    error('maxwell_results.mat has %d unknowns but rebuilt geometry has N = %d; params mismatch, re-solve.', numel(Jx), N);
end

x0 = [0.0; 0.0; 2.5];
p0 = [1.0; 1.0; 1.0];
dr = (1 - lam_inner)/nch1;


ng = 50;
g = linspace(-1, 1, ng);
[X, Y] = meshgrid(g, g);
lam_g = sqrt(X.^2 + Y.^2);
sel = lam_g <= 1;
eval_xyz = [X(sel).'; Y(sel).'; zeros(1, nnz(sel))];
nt = size(eval_xyz, 2);

lam_e = sqrt(eval_xyz(1, :).^2 + eval_xyz(2, :).^2);
t_e = mod(atan2(eval_xyz(2, :), eval_xyz(1, :)), 2*pi).';


[pid, uvs] = find_patch_uv(inner_src, eval_xyz);
targinfo_eval = struct('r', eval_xyz, 'patch_id', pid, 'uvs_targ', uvs);
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

[S_Jx, dxJx, ~] = efie2_eval(Jx, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[S_Jy, ~, dyJy] = efie2_eval(Jy, inner_src, D_J, eval_xyz, zk, i2e_S, i2e_grad, QeJ);
[S_rho, dx_rho, dy_rho] = efie2_eval(rho, inner_src, D_rho, eval_xyz, zk, i2e_S, i2e_grad, QeR);

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


dens = {Jx, Jy, rho};
Dband = {D_J, D_J, D_rho};
names = {'$J_1$', '$J_2$', '$\rho$'};
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
