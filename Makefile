DOCKER = exec docker run --rm -i --user="$$(id -u):$$(id -g)" --net=none -v "$$PWD":/data "blang/latex:ubuntu"

TARGETS = circllhist.pdf

all: fix $(TARGETS)

%.pdf: %.tex
	$(DOCKER) pdflatex -interaction=nonstopmode --shell-escape $<

clean:
	rm $(TARGETS) || true

docker-clean:
	docker rmi "blang/latex:ubuntu"

fix:
	cd evaluation/images && rename -v -f 's/ /_/' *.png
	cd evaluation/tables && rename -v -f 's/ /_/' *.tex

open:
	open $(TARGETS)
