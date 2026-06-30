Project guardrails for ObserverBench:

This repo is a reproduction and diagnostic workbench for the ObserverBench paper. It is not a new research project and not a general benchmark platform yet.

Do not:
- add new experiments,
- add new tasks beyond the current paper,
- change scientific claims,
- invent missing results,
- broaden the scope into a hosted demo, product, general plugin platform, or InterpBench sweep,
- relabel weak/null results as positive wins.

Definition of done for v0:
- reproduce the paper figures/tables from frozen outputs,
- run quick smoke tests on CPU,
- generate ObserverCards from metrics,
- expose the current tasks through a stable-enough internal CLI,
- document exact command-to-figure mapping,
- pass tests that pin the scientific claims.

Stop when the repo matches the paper. Do not expand the paper.
