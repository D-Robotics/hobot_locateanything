.PHONY: test test-fast prepare calibrate build verify compile-language compile-vision clean-python-cache

PYTHON ?= python

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest tests -q

prepare:
	$(PYTHON) compiler/quantize.py prepare

calibrate:
	$(PYTHON) compiler/quantize.py calibrate

build:
	$(PYTHON) compiler/quantize.py build --component all --target hbm

verify:
	$(PYTHON) compiler/quantize.py verify --component all --level all

compile-language:
	$(PYTHON) compiler/quantize.py build --component language --target hbm

compile-vision:
	$(PYTHON) compiler/quantize.py build --component vision --target hbm

clean-python-cache:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc') if '.git' not in p.parts]"
