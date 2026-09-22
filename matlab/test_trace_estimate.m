function test_trace_estimate
% Public contract, finite caps, and representation-independent probe law.
    for precision={'double','single'}
        lam=cast(logspace(0,-4,96)',precision{1});
        for method={'scalar','block','scalar_auto','block_auto'}
            args={'Method',method{1},'TargetRelError',1e-3, ...
                  'MaxMatvecs',400,'Depth',20,'Seed',77};
            [e,t]=randlapack.trace_estimate(lam,args{:});
            [ed,td]=randlapack.trace_estimate(diag(lam),args{:});
            assert(isfinite(e) && t.matvecs<=400);
            assert(t.matvecs==td.matvecs);
            assert(abs(e-ed)<100*eps(precision{1})*max(abs(e),1));
            assert(~t.total_error_certified);
            assert(t.oracle_mean_depth <= t.depth_limit + 1e-9);
            assert(abs(t.oracle_mean_depth*t.samples + t.rank+t.probe_mv-t.matvecs)<1e-8);
            assert(t.oracle_mean_depth==td.oracle_mean_depth);
        end
        for method={'scalar_auto','block_auto'}
            for fraction=[1 .5 .25 .125]
                for qfraction=[1 .25]
                    [e,~,~,t]=randlapack.fun_nystrom_pp(lam,1,1, ...
                        'LFAType',method{1},'Budget',180,'AutoEps',1e-3, ...
                        'CapRankFraction',fraction,'QuadratureFraction',qfraction);
                    assert(isfinite(e) && t.probe_mv+t.auto_k+t.oracle_mv<=180);
                end
            end
            for cap=[1 16 17 32 100]
                try
                    [e,t]=randlapack.trace_estimate(lam,'Method',method{1}, ...
                        'MaxMatvecs',cap,'TargetRelError',1e-3);
                    assert(isfinite(e) && t.matvecs<=cap && cap>=17);
                catch ME
                    assert(cap<17 && contains(ME.identifier,'infeasibleBudget'),ME.message);
                end
            end
        end
    end
    for method={'scalar_auto','block_auto'}
        try
            randlapack.fun_nystrom_pp(lam,1,1,'LFAType',method{1}, ...
                'Func','entropy','Budget',100);
            error('test_trace_estimate:acceptedEntropy','Unsupported certificate accepted.');
        catch ME
            assert(contains(ME.identifier,'unsupportedAutoFunction'),ME.message);
        end
    end
    % Canonicalization must happen before the function-specific Lambda branch.
    A=logspace(0,-3,48)';
    for method={'scalar','block_auto'}
        args={'Method',method{1},'Lambda',.25,'Depth',8,'MaxMatvecs',96,'Seed',7};
        [a,ta]=randlapack.trace_estimate(A,args{:},'Func','effdim');
        [b,tb]=randlapack.trace_estimate(A,args{:},'Func','EFFDIM');
        assert(a==b && ta.matvecs==tb.matvecs && strcmp(tb.function_name,'effdim'));
    end
    fprintf('Four-method target/cap and dense/diagonal tests passed in both precisions.\n');
end
