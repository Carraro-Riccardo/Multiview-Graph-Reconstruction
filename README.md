# Graph Reconstruction

Ricostruzione multi-vista di grafi idraulici a partire da immagini con simboli
Shurikode. Il progetto rileva e decodifica i componenti, riconcilia le istanze
visibili in immagini diverse, usa la geometria VGGT come supporto 3D e produce
un grafo finale valutabile, visualizzabile ed esportabile in QGIS.

## Comandi principali

Eseguire i comandi dalla directory principale del progetto.

```powershell
# Ricostruzione di tutti i nodi trovati in IMAGES_DIR
python run.py

# Ricostruzione di nodi specifici
python run.py 2 7 15_1

# Valutazione di tutti i risultati disponibili
python evaluate.py

# Valutazione di nodi specifici
python evaluate.py 2 7 9

# Visualizzazione del grafo ricostruito
python visualize.py 2
python visualize.py 2 7 9
python visualize.py --all

# Esportazione QGIS batch di tutti i fused_graph.json
python json_adapter_to_qgis.py
```

Su Windows è disponibile anche il flusso completo ricostruzione → valutazione
→ visualizzazione:

```powershell
.\tools\run_pipeline.cmd
.\tools\run_pipeline.cmd 2 7 9
```

## Configurazione

La configurazione locale vive in `.env`, che non viene versionato. Per creare
una configurazione nuova:

```powershell
Copy-Item .env.template .env
```

I percorsi relativi sono sempre risolti rispetto alla directory principale del
progetto, non rispetto alla directory dalla quale Python viene avviato.

| Variabile | Obbligatoria | Significato |
| --- | --- | --- |
| `IMAGES_DIR` | sì | Directory delle immagini `node<id>_<frame>.jpg`. |
| `LABELS_EXT_DIR` | sì | Annotazioni estese per la valutazione delle detection. |
| `YOLO_MODEL_PATH` | sì | File dei pesi YOLO. |
| `ASSETS_DIR` | no | Asset Shurikode; default `./assets`. |
| `ANNOTATIONS_DIR` | no | Grafi ground truth; default `./annotations`. |
| `WORK_DIR` | no | Root degli output; default `./output`. |
| `DEBUG_OUTPUT_ENABLED` | no | Genera dump interni della pipeline; default `false`. |
| `DEBUG_OUTPUT_DIR` | no | Root della diagnostica; default `./output/debug`. |
| `VGGT_EDGE_ENABLED` | no | Abilita supporto geometrico VGGT; default `true`. |
| `VGGT_EDGE_SEARCH_GLOB` | no | Nome della directory/risorsa VGGT cercata. |
| `VGGT_GENERATE_IF_MISSING` | no | Genera gli artefatti VGGT mancanti. |
| `VGGT_EMBEDDED_PYTHON` | no | Python dedicato a VGGT; default `./python312/python.exe`. |
| `VGGT_MODEL_ID` | no | Modello Hugging Face usato da VGGT. |
| `VGGT_PREPROCESS_MODE` | no | Pre-processing VGGT, normalmente `crop`. |

Le variabili `VGGT_3D_ADJACENCY_FACTOR`, `VGGT_3D_MUTUAL_TOP_K`,
`VGGT_3D_MIN_POINTS_PER_OBSERVATION` e
`VGGT_3D_MAX_POINTS_PER_OBSERVATION` espongono i parametri geometrici avanzati;
i valori consigliati sono già presenti in `.env.template`.

Le soglie algoritmiche, le valenze, i codici dei componenti e i prior
strutturali sono definiti in
[`graph_reconstruction/config.py`](graph_reconstruction/config.py).

## Ambiente Python

`run.py`, `evaluate.py`, `visualize.py` e l’adattatore QGIS vanno eseguiti con
l’ambiente Python principale del progetto. Le dipendenze principali sono
NumPy, SciPy, OpenCV, Pillow, NetworkX, Matplotlib, Ultralytics e il decoder
Shurikode.

Setup automatico su Windows:

```powershell
# Dipendenze pubbliche + verifica del decoder già installato
.\setup_environment.ps1

# Oppure installazione del decoder privato da wheel/cartella sorgente
.\setup_environment.ps1 -ShurikodePackagePath C:\path\to\shurikodecontext.whl

.\.venv\Scripts\Activate.ps1
```

Le dipendenze pubbliche sono elencate in `requirements.txt`. Il decoder
specifico del progetto, che espone `from shurikodecontext import Dec`, non è
incluso tra queste dipendenze e va fornito separatamente al setup.

La directory `python312/` è il runtime embedded usato internamente per VGGT:
non è il punto di ingresso normale del progetto.

## Struttura del repository

```text
GraphReconstruction/
├── run.py                         # ricostruzione
├── evaluate.py                    # metriche e confronto standard
├── visualize.py                   # rendering del grafo finale
├── json_adapter_to_qgis.py        # adattatore GeoJSON per QGIS
├── .env.template                  # configurazione di esempio
├── annotations/                   # grafi ground truth
├── assets/                        # cataloghi e simboli Shurikode
├── docs/                          # documentazione tecnica
├── evaluation/
│   └── standard_comparison/       # confronto con i grafi di riferimento
├── graph_reconstruction/
│   ├── config.py                  # configurazione e prior
│   ├── pipeline/                  # pipeline attiva
│   ├── shared/                    # detection, JSON e assignment
│   └── visualization/             # bird view e simboli Shurikode
├── tools/
│   ├── debug/                     # diagnostica manuale
│   ├── examples/                  # dimostrazioni algoritmiche
│   ├── qgis/                      # packaging degli export QGIS
│   ├── reporting/                 # figure specialistiche per report
│   └── run_pipeline.cmd           # comando Windows end-to-end
├── output/                        # risultati e cache generati
└── python312/                     # runtime embedded VGGT
```

### Entry point

La root contiene intenzionalmente soltanto quattro file Python eseguibili:

- `run.py` scopre i nodi e avvia la ricostruzione;
- `evaluate.py` calcola metriche su istanze e archi e confronta gli standard;
- `visualize.py` genera `fused_graph_plot.png`;
- `json_adapter_to_qgis.py` converte il grafo in GeoJSON.

### Package `graph_reconstruction`

- `config.py`: carica `.env`, normalizza i path e contiene prior e soglie;
- `pipeline/orchestrator.py`: coordina l’intero flusso;
- `pipeline/observed_graphs.py`: grafi osservati per singola vista;
- `pipeline/node_reid.py`: riconcilia le detection tra immagini;
- `pipeline/vggt_*`: generazione e uso della geometria 3D;
- `pipeline/direct_graph_builder.py`: costruzione degli archi;
- `pipeline/graph_refinement.py`: connettività e serializzazione;
- `pipeline/diameter_annotation.py`: diametri sugli archi;
- `shared/detection.py`: YOLO, decoder Shurikode e cache;
- `shared/io.py`: scrittura JSON;
- `shared/linear_assignment.py`: Hungarian matching con fallback;
- `visualization/`: rendering riutilizzato dalla pipeline e da QGIS.

Per il dettaglio algoritmico vedere [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Formato degli input

Le immagini devono rispettare il pattern:

```text
node<node_id>_<frame_numerico>.jpg
```

Esempi validi:

```text
node2_0001.jpg
node17_2_0042.jpg
node15_1_0003.jpg
```

Il ground truth del grafo usa `ANNOTATIONS_DIR/node<id>_gt.json`. Le label
estese delle immagini sono lette da `LABELS_EXT_DIR` durante la valutazione.

## Output

Gli output normali sono separati dalle cache e dalla diagnostica:

```text
output/
├── detection_cache/               # cache YOLO/decoder per immagine
├── multiview/
│   └── node<id>/
│       ├── fused_graph.json        # risultato principale
│       ├── scene_nodes.json        # istanze multi-vista
│       ├── bird_view.png           # vista 3D proiettata
│       ├── fused_graph_plot.png    # creato da visualize.py
│       ├── vggt_geometry.json      # geometria/cache VGGT
│       └── vggt/*.npz              # artefatti VGGT costosi
├── standard_comparison/            # tabelle e ranking di evaluate.py
└── debug/                           # presente solo quando richiesto
```

`vggt_edge_debug.json` può comparire nei risultati creati prima del refactoring:
viene ancora letto per compatibilità, mentre i nuovi run usano
`vggt_geometry.json`.

Per evitare output dispersivi, i dump ReID, diametri e richieste VGGT vengono
scritti soltanto con:

```dotenv
DEBUG_OUTPUT_ENABLED=true
```

La guida agli strumenti diagnostici è in
[`tools/debug/README.md`](tools/debug/README.md).

## Valutazione e standard

`evaluate.py` usa tre sorgenti:

1. detection e scene node prodotti dalla pipeline;
2. label estese in `LABELS_EXT_DIR`;
3. grafi annotati in `ANNOTATIONS_DIR`.

Il confronto con i dodici standard di riferimento è implementato in
`evaluation/standard_comparison/` e produce tabelle aggregate sotto
`output/standard_comparison/`. `visualize.py` genera inoltre automaticamente:

```text
output/standard_comparison/report_completo.html
```

Il report è autosufficiente e mostra, nello stesso file HTML, il grafo
ricostruito a sinistra e il relativo best match standard a destra. Con
`visualize.py --all` il singolo documento contiene tutti i nodi disponibili.

## QGIS

Conversione batch con geometria sintetica:

```powershell
python json_adapter_to_qgis.py
```

Conversione di un singolo grafo usando la geometria VGGT quando disponibile:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --geometry-source auto
```

Packaging degli export combinati:

```powershell
python tools/qgis/package_qgis_geojson.py --overwrite
```

Opzioni, layout e attributi sono descritti in [`docs/QGIS.md`](docs/QGIS.md).

## Flusso di lavoro consigliato

```powershell
# 1. Configurazione
Copy-Item .env.template .env

# 2. Ricostruzione mirata durante lo sviluppo
python run.py 2

# 3. Valutazione
python evaluate.py 2

# 4. Rendering
python visualize.py 2

# 5. Esportazione QGIS, se necessaria
python json_adapter_to_qgis.py output/multiview/node2/fused_graph.json --geometry-source auto
```

Le cache rendono i run successivi più rapidi. Per indagare un problema ReID o
geometrico, abilitare gli output diagnostici, rieseguire il nodo interessato e
usare uno degli script documentati in `tools/debug/README.md`.
