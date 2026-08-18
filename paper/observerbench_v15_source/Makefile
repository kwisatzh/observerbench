# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

PYTHON ?= python3

.PHONY: all artifacts phase04 phase05-nonlinear phase06-ioi phase07-ioi phase08-ioi safety-figure final phase01 clean

all: final

artifacts: phase04 phase05-nonlinear phase06-ioi phase07-ioi phase08-ioi safety-figure

phase04:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/build_phase04_artifact.py
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/reproduce_paper_fast.py --outdir paper/generated_current

phase05-nonlinear:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/build_phase05_nonlinear_suffix_artifact.py

phase06-ioi:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/build_phase06_ioi_confirmation_artifact.py

phase07-ioi:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/build_phase07_ioi_confirmation_artifact.py

phase08-ioi:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/build_phase08_ioi_sensitivity_artifact.py

safety-figure:
	cd ../.. && PYTHONPATH=src MPLBACKEND=Agg $(PYTHON) scripts/plot_qwen_safety_score_calibration.py --rows-root results/revision/qwen_safety/paired_scope_v0/followup_v0/observer_rows --output paper/observerbench_v15_source/figures/qwen_safety_stratum_calibration.png

final: artifacts
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build observerbench.tex

phase01:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build observerbench_phase01.tex

clean:
	latexmk -C -outdir=build observerbench.tex
	latexmk -C -outdir=build observerbench_phase01.tex
