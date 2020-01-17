DOCKER = exec docker run --rm -i --user="$$(id -u):$$(id -g)" --net=none -v "$$PWD":/data "blang/latex:ubuntu"

TARGETS = circllhist.pdf
ENV = LC_ALL=C

all: fix $(TARGETS)

%.pdf: %.tex
	$(DOCKER) pdflatex -interaction=nonstopmode $<

%.tex: %.md
	pandoc -o $@ $<

clean:
	rm $(TARGETS) || true

docker-clean:
	docker rmi "blang/latex:ubuntu"

fix:
	cd evaluation/images && $(ENV) rename -v -f 's/ /_/' *.png
	cd evaluation/tables && $(ENV) rename -v -f 's/ /_/' *.tex

open:
	open $(TARGETS)
