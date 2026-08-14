function state = rl_normalize_state(in, caller)
%RL_NORMALIZE_STATE  Coerce a user-supplied RNG state into the MEX form.
%
%   Accepts a struct with .counter (uint32[4]) and .key (uint32[2]), or a
%   scalar numeric seed (placed in key(2), with the counter left at zero).
%   CALLER names the user-facing function, so the raised error identifier
%   and message point at what the user actually called.

    if isstruct(in)
        if ~isfield(in, 'counter') || ~isfield(in, 'key')
            error(sprintf('randlapack:%s:state', caller), ...
                  'state struct must have fields .counter and .key');
        end
        validateattributes(in.counter, {'uint32'}, {'vector', 'numel', 4}, ...
                           caller, 'state.counter');
        validateattributes(in.key,     {'uint32'}, {'vector', 'numel', 2}, ...
                           caller, 'state.key');
        state.counter = uint32(in.counter(:)).';
        state.key     = uint32(in.key(:)).';
    elseif isnumeric(in) && isscalar(in)
        % Treat a scalar as a seed; place it in key(2). Counter starts at zero.
        state.counter = uint32([0, 0, 0, 0]);
        state.key     = uint32([0, uint32(in)]);
    else
        error(sprintf('randlapack:%s:state', caller), ...
              'state must be a struct or a scalar seed');
    end
end
