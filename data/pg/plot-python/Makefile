
echo:
	echo "Specify a command: <build> or <run>."

build:
	docker build -t avg-dist-plot-python:3.13 .

run:
	docker run -it --rm -v ./../result:/usr/src/app/result avg-dist-plot-python
