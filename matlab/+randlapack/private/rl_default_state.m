function state = rl_default_state()
%RL_DEFAULT_STATE  Default Philox4x32 RNG state (all zeros).
%
%   Shared by the randlapack.* wrappers. Functions in a private/ directory
%   are visible to the enclosing package's functions and to nobody else,
%   which is exactly the scope these helpers want. The compiled MEX files
%   land in this same directory, for the same reason.

    state.counter = uint32([0, 0, 0, 0]);
    state.key     = uint32([0, 0]);
end
