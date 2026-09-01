# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

.PHONY: demo demo-safety demo-qwen

demo: demo-safety demo-qwen

demo-safety:
	@python3 demo/safety_tutorial.py

demo-qwen:
	@python3 practice/qwen_copy_v2_b040/baseline.py --output /tmp/observerbench-qwen-practice-predictions.csv
	@python3 practice/qwen_copy_v2_b040/score.py /tmp/observerbench-qwen-practice-predictions.csv
