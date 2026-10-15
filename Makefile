# Placeholder targets; filled in as each phase lands.
.PHONY: test data baselines dl agents live
test:
	python -m pytest tests -q
data:
	@PYTHONPATH=src python -m oilfc.data.build
baselines:
	@echo "TODO Phase 2: python experiments/run_baselines.py"
dl:
	@echo "TODO Phase 3-4: python experiments/run_dl.py"
agents:
	@echo "TODO Phase 4b: python experiments/run_agents.py"
live:
	@echo "TODO: daily live forward test (frozen v1)"
