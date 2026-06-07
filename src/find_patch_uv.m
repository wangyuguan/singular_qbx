function [pid, uvs] = find_patch_uv(S, tgt)
[~, srccoefs, norders, ixyzs, ~, ~] = extract_arrays(S);
norder = norders(1);
npat = S.npatches;
npts = S.npts;
nev = size(tgt, 2);
pid = -1*ones(nev, 1);
uvs = zeros(2, nev);

nodes = S.r;
uvn = S.uvs_targ;
patch_of_node = zeros(npts, 1);
for p = 1:npat
    patch_of_node(ixyzs(p):ixyzs(p+1)-1) = p;
end

for a = 1:nev
    x = tgt(:, a);
    d2 = sum((nodes - x).^2, 1);
    [~, near] = mink(d2, min(16, npts));
    cand = unique(patch_of_node(near), 'stable');
    best_in = -inf;
    best_p = -1;
    best_uv = [0; 0];
    for c = 1:numel(cand)
        p = cand(c);
        cols = ixyzs(p):ixyzs(p+1)-1;
        coefs = srccoefs(1:3, cols);
        % seed Newton from this patch's node nearest to the target
        [~, jloc] = min(sum((nodes(:, cols) - x).^2, 1));
        uv = uvn(:, cols(jloc));
        ok = false;
        for it = 1:30
            [pl, du, dv] = koorn.ders(norder, uv);
            Xp = coefs*pl;
            res = Xp(1:2) - x(1:2);
            if norm(res) < 1e-13
                ok = true;
                break;
            end
            J = [coefs(1:2, :)*du, coefs(1:2, :)*dv];
            uv = uv - J\res;
        end
        % keep the most-interior valid fit so seam targets are assigned consistently
        if ok
            interior = min([uv(1), uv(2), 1 - uv(1) - uv(2)]);
            if interior > best_in
                best_in = interior;
                best_p = p;
                best_uv = uv;
            end
        end
    end
    if best_in >= -1e-7
        pid(a) = best_p;
        uvs(:, a) = best_uv;
    end
end
end
