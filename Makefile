DOCKER = exec docker run --rm -i --user="$$(id -u):$$(id -g)" --net=none -v "$$PWD":/data "blang/latex:ubuntu"

TARGETS = circllhist.pdf

all: $(TARGETS)

%.pdf: %.tex
	$(DOCKER) pdflatex -interaction=nonstopmode --shell-escape $<

clean:
	rm $(TARGETS) || true

docker-clean:
	docker rmi "blang/latex:ubuntu"

open:
	open $(TARGETS)