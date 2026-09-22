function test_trace_effdim_parameter
% Public and expert interfaces must estimate the SAME shifted function.
    lam=logspace(-2,2,80)';
    for shift=[.1 1 10]
        for method={'scalar','block','scalar_auto','block_auto'}
            [e,t]=randlapack.trace_estimate(lam,'Method',method{1},'Func','effdim', ...
                'Lambda',shift,'Depth',10,'MaxMatvecs',200,'Seed',91);
            if endsWith(method{1},'_auto')
                expected=randlapack.fun_nystrom_pp(lam,1,1,'LFAType',method{1}, ...
                    'Func','effdim','PolyLambda',shift,'Budget',200,'AutoEps',1e-3, ...
                    'CapRankFraction',.5,'SketchSeed',91);
            else
                expected=randlapack.fun_nystrom_pp(lam,t.rank,t.samples,'LFAType',[method{1} '_qfa'], ...
                    'Func','effdim','PolyLambda',shift,'Depth',10,'Adaptive',0,'Q',1,'SketchSeed',91);
            end
            assert(abs(e-expected)<1e-12*max(1,abs(expected)));
        end
    end
    f=@(x) x./(x+1);g=@(x) x./(x+10);
    assert(abs(sum(f(lam))-sum(g(lam)))>1,'Test must distinguish the two regularizations.');
    fprintf('Effective-dimension parameter forwarding passed for all four methods.\n');
end
