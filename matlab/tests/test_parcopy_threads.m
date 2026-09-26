function test_parcopy_threads()
% The threaded input copy must never take MATLAB down: with an absurd OMP_NUM_THREADS it has to
% clamp or fall back and still return the same estimate as the plain copy.
    rng(3); n = 1024; [Q,~] = qr(randn(n)); A = Q*diag(logspace(0,-4,n))*Q'; A = (A+A')/2;
    args = {'Func','log','LFAType','block_qfa','Depth',10,'Q',1,'SketchSeed',5,'SkipSymCheck',true};
    setenv('RANDLAPACK_PERF_PARCOPY','0'); e0 = randlapack.fun_nystrom_pp(A, 64, 16, args{:});
    setenv('RANDLAPACK_PERF_PARCOPY','1'); setenv('OMP_NUM_THREADS','200000');
    e1 = randlapack.fun_nystrom_pp(A, 64, 16, args{:});
    setenv('OMP_NUM_THREADS','4'); setenv('RANDLAPACK_PERF_PARCOPY','0');
    assert(e1 == e0, 'estimate changed: %.17g vs %.17g', e1, e0);
    fprintf('TEST_PARCOPY_THREADS PASSED (%.17g)\n', e1);
end
