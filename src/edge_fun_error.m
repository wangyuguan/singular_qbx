function err = edge_fun_error(D, dens)

k = D.k;
nch1 = D.nch1;
nch2 = D.nch2;
nf = size(dens, 2);
err = zeros(nch1, nch2);
for j = 1:nch2
    for i = 1:nch1
        ind = ((j-1)*nch1 + (i-1))*k^2 + (1:k^2);
        e = 0;
        for f = 1:nf
            V = reshape(dens(ind, f), k, k);
            C = D.v2c_lam(:, :, i) * V * D.v2c_t(:, :, j).';
            e = max(e, max(max(abs(C(k, :))), max(abs(C(:, k)))));
        end
        err(i, j) = e;
    end
end
end
