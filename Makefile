# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

.PHONY: demo demo-safety demo-qwen demo-integrity demo-relay

demo: demo-safety demo-qwen

demo-safety:
	@python3 demo/safety_tutorial.py

demo-integrity:
	@PYTHONPATH=src python3 examples/evidence_integrity_submission.py

demo-relay:
	@PYTHONPATH=src python3 examples/evidence_integrity_submission.py --variant cross-agent-origin

demo-qwen:
	@python3 practice/qwen_copy_v2_b040/baseline.py --output /tmp/observerbench-qwen-practice-predictions.csv
	@python3 practice/qwen_copy_v2_b040/score.py /tmp/observerbench-qwen-practice-predictions.csv
