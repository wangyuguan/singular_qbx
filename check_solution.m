
clear
clc

run('../fmm3dbie_hirax/matlab/startup.m')
addpath('../FMM3D/matlab/')
addpath('../chebfun')
addpath('src/')

load('maxwell_results.mat', 'Jx', 'Jy', 'rho')


lam_inner = 0.9;
k = 8;
nch1 = 2;
nch2 = 24;
gamma = @(t) [cos(t); sin(t); 0];
dgamma = @(t) [-sin(t); cos(t); 0];
t_splits = linspace(0, 2*pi, nch2+1);

D_rho = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, -1/2);
D_J = set_edge_patch(k, nch1, nch2, lam_inner, gamma, dgamma, t_splits, 1/2);
nb = size(D_rho.src_xyz, 2);

norder = k;
inner_src = geometries.disk([1, 1], [], [4 4 4], norder);
inner_src = scale(inner_src, lam_inner);
ni = inner_src.npts;


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
