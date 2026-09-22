function test_trace_sketch_density
% Verify that the public knob changes the actual head, including auto modes.
    A=logspace(0,-4,128)';
    for method={'scalar','block','scalar_auto','block_auto'}
        estimates=[];
        for vnz=[8 16 32 0]
            [e,t]=randlapack.trace_estimate(A,'Method',method{1},'Depth',8, ...
                'MaxMatvecs',256,'Seed',701,'VecNnz',vnz);
            args={'Func','sqrt','VecNnz',vnz,'SketchSeed',701};
            if endsWith(method{1},'_auto')
                ref=randlapack.fun_nystrom_pp(A,1,1,args{:},'LFAType',method{1}, ...
                    'Budget',256,'AutoEps',1e-3,'CapRankFraction',.5);
            else
                ref=randlapack.fun_nystrom_pp(A,t.rank,t.samples,args{:}, ...
                    'LFAType',[method{1} '_qfa'],'Q',1,'Depth',8,'Adaptive',0);
            end
            assert(abs(e-ref)<1e-12*max(1,abs(ref)));
            assert(t.vec_nnz_effective>=1 && t.vec_nnz_effective<=t.rank);
            estimates(end+1)=e; %#ok<AGROW>
        end
        assert(max(estimates)-min(estimates)>1e-8,'Density control did not affect this head.');
    end
    try
        randlapack.fun_nystrom_pp(A,8,8,'Sketch','gaussian');
        error('test_trace_sketch_density:acceptedUnsupported','Unsupported sketch was accepted.');
    catch ME
        assert(strcmp(ME.identifier,'randlapack:fun_nystrom_pp:unsupportedSketch'),ME.message);
    end
    fprintf('Sketch-density forwarding/effect and unsupported-sketch rejection passed.\n');
end
