

.PHONY: report clean

BIBLIOGRAPHY = bibliography.bib
MARKDOWN_FILE = report.md
PDF_FILE = report.pdf
PANDOCFLAGS =                         \
  --from=markdown                     \
  --highlight-style=pygments          \
  --pdf-engine=xelatex                \
  --citeproc                          \
  --bibliography=$(BIBLIOGRAPHY)      

$(PDF_FILE): $(MARKDOWN_FILE) $(BIBLIOGRAPHY)
	pandoc $< -o $@ $(PANDOCFLAGS)

report: $(PDF_FILE)

clean:
	$(RM) $(PDF_FILE)

