.PHONY: all data test cv sens loso shift cross report paper clean
PY=CKINN_THREADS=1 python
all: data test cv sens loso shift cross report paper
data:
	$(PY) scripts/00_harmonize.py data/raw/RCHealthCareData-20180820.xlsx data/raw/DatasetIntJObesity.xlsx data/processed
test:
	python -m pytest -q tests
cv sens loso shift cross:
	$(PY) scripts/run_experiments.py $@
report:
	python scripts/make_report.py
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
