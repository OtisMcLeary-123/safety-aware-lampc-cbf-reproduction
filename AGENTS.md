# Agent ownership

Five specialist agents own separate parts of the reproduction:

1. Python integration: package layout, typed configuration, CLI, orchestration, and unit-test fixtures.
2. do-mpc: discrete robot model, horizon, objective, bounds, and receding-horizon controller setup.
3. CasADi: symbolic objective terms, obstacle barrier function, and discrete CBF inequality.
4. IPOPT: nonlinear solver options, solve diagnostics, feasibility handling, and solver smoke tests.
5. Safe Panda Gym: environment adapter, observation/action mapping, dynamic obstacle sensing, and simulation smoke test.

Agents must not edit another specialist's owned module without coordinating through the integration interfaces. Do not commit generated results, credentials, API keys, environments, or the source PDF.

