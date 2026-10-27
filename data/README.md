# Dati di prova

La repository mantiene la struttura attesa dalla pipeline, ma non include le
immagini né le label estese del dataset.

- `images/`: immagini JPG nominate `node<id>_<frame>.jpg`, senza sottocartelle;
- `labels_extended/`: una label `<image_stem>_extended.txt` per ogni immagine;
- `graph_annotations/`: grafi ground truth `node<id>_gt.json` usati dalla valutazione.

Per una prova su un solo nodo è sufficiente copiare in `images/` tutte le sue
viste e in `labels_extended/` i file corrispondenti. I percorsi possono essere
sostituiti nel file `.env` quando il dataset viene conservato altrove.
