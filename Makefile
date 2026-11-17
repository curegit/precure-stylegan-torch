.PHONY: compile format test

compile:
	python3 -m compileall -f -l .
	python3 -m compileall -f stylegan interface utilities

format:
	python3 -m black -l 500 --target-version py314 .

test:
	python3 show.py models/mnist.pt
	python3 generate.py models/mnist.pt -n 10 -l seven -o output-mnist
