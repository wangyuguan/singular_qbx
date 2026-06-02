function segs = generate_dyadic_segments(nch, a, b, refine_left)
segs = (b-a)*(.5).^(1:nch-1);
if refine_left
    segs = a+segs;
else
    segs = b-segs;
end
segs = [segs,a,b];
segs = sort(segs);
end
