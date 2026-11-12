.PHONY: format

format:
	python3 -m black -l 500 --target-version py314 .
