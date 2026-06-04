PDF = cutile_hpl_ai_porting_report.pdf
TEX = cutile_hpl_ai_porting_report.tex
BIB = cutile_hpl_ai_porting_report.bib
BUILD = build
BASE = cutile_hpl_ai_porting_report

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	mkdir -p $(BUILD)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(TEX)
	bibtex $(BUILD)/$(BASE)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) $(TEX)
	cp $(BUILD)/$(PDF) $(PDF)

clean:
	rm -rf $(BUILD) $(PDF)
