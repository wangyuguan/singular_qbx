clear; clc;
load('maxwell_disk_results.mat')

th = linspace(0, 2*pi, 200);

figure 
tiledlayout(1,3)
nexttile 
hold on 
scatter(src_xyz_J(1, :), src_xyz_J(2, :), 15, real(J_u), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar; 
title('Re($J_1$)', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);

nexttile  
hold on 
scatter(src_xyz_J(1, :), src_xyz_J(2, :), 15, real(J_v), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar; 
title('Re($J_2$)', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);


nexttile 
hold on
scatter(src_xyz_rho(1, :), src_xyz_rho(2, :), 15, real(rho), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar;
title('Re($\rho$)', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);



figure 
tiledlayout(1,3)
nexttile 
hold on 
scatter(src_xyz_J(1, :), src_xyz_J(2, :), 15, log10(abs(J_u)), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar; 
title('$\log_{10}|J_1|$', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);

nexttile  
hold on 
scatter(src_xyz_J(1, :), src_xyz_J(2, :), 15, log10(abs(J_v)), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar; 
title('$\log_{10}|J_2|$', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);


nexttile 
hold on
scatter(src_xyz_rho(1, :), src_xyz_rho(2, :), 15, log10(abs(rho)), 'filled');
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
colorbar;
title('$\log_{10}|\rho|$)', Interpreter='latex', FontSize=16); 
xlabel('$x$', Interpreter='latex', FontSize=16); 
ylabel('$y$', Interpreter='latex', FontSize=16);




figure 
hold on;
plot(cos(th), sin(th), 'k-');
plot(lam_inner*cos(th), lam_inner*sin(th), 'k:');
scatter(eval_xyz(1, :), eval_xyz(2, :), 80, log10(err), 'filled');
for q = 1:numel(err)
    text(eval_xyz(1, q), eval_xyz(2, q), sprintf('  %.1e', err(q)), 'FontSize', 8);
end
