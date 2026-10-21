# Manuale tecnico LaTeX

Il documento principale e' `main.tex`. Tutti i percorsi sono relativi a questa
directory.

Compilazione consigliata:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

In alternativa, eseguire `pdflatex` almeno due volte per risolvere indice,
riferimenti e numero totale delle pagine.

Il PDF verificato e versionato e' `main.pdf`, nella stessa directory del
sorgente.

## Figure

Le figure raster incluse nel documento sono in `figures/`; i diagrammi di
flusso sono sorgenti TikZ. Il registro sintetico e' in `figures/README.md`.
