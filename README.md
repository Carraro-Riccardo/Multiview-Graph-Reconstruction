# Graph Reconstruction

Ricostruzione multi-vista di grafi idraulici a partire da immagini con simboli
Shurikode. Il progetto rileva e decodifica i componenti, riconcilia le istanze
visibili in immagini diverse, usa la geometria VGGT come supporto 3D e produce
un grafo finale valutabile, visualizzabile ed esportabile in QGIS.

## Installazione

Eseguire i comandi dalla root del repository. Creare l'ambiente Conda
versionato e attivarlo:

```powershell
conda env create -f environment.yml
conda activate graph_rec
```

Per aggiornare un ambiente esistente:

```powershell
conda env update -n graph_rec -f environment.yml --prune
```

Il file riproduce lo stack Linux/CUDA validato dal benchmark e non è
direttamente portabile su Windows, perché include pacchetti di sistema Linux.

Il decoder Shurikode va installato dal repository dedicato:

```powershell
git clone https://github.com/Carraro-Riccardo/Shurikode-Context.git
python -m pip install ./Shurikode-Context
```

Installare infine la revisione VGGT usata nello sviluppo:

```powershell
python -m pip install "git+https://github.com/facebookresearch/vggt.git@44b3afbd1869d8bde4894dd8ea1e293112dd5eba"
```

Al primo utilizzo VGGT scarica i pesi del modello; il primo run richiede quindi
una connessione Internet e può essere sensibilmente più lento.

## Configurazione

La configurazione della macchina e della lavorazione vive in `.env`, che non
viene versionato. Crearlo dal template:

```powershell
Copy-Item .env.template .env
```

Impostare almeno:

```dotenv
IMAGES_DIR=C:/percorso/dataset/images
LABELS_EXT_DIR=C:/percorso/dataset/labels_extended
YOLO_MODEL_PATH=./models/modello.pt
WORK_DIR=./output
ASSETS_DIR=./assets
ANNOTATIONS_DIR=C:/percorso/dataset/graph_annotations
DEBUG_OUTPUT_ENABLED=false
DEBUG_OUTPUT_DIR=./output/debug
```

`IMAGES_DIR`, `LABELS_EXT_DIR` e `YOLO_MODEL_PATH` sono obbligatorie. Le label
e le annotazioni servono alla valutazione, non alla ricostruzione. I percorsi
relativi sono risolti rispetto alla root del repository.

Le opzioni VGGT facoltative sono già documentate in `.env.template`. Se viene
usato un runtime dedicato, `VGGT_EMBEDDED_PYTHON` deve indicarne l'interprete.
Soglie, codici, valenze e vincoli strutturali sono invece definiti in
[`graph_reconstruction/config.py`](graph_reconstruction/config.py).

## Primo avvio

Iniziare da un solo nodo e controllare il risultato prima del batch:

```powershell
conda activate graph_rec
python run.py 7
python evaluate.py 7
python visualize.py 7
```

Il risultato principale è `WORK_DIR/multiview/node7/fused_graph.json`; il
render è `fused_graph_plot.png`. La valutazione supervisionata viene eseguita
solo quando sono disponibili label estese e annotazione del grafo.

Per elaborare tutti gli ID scoperti:

```powershell
python run.py
python evaluate.py
python visualize.py --all
```

Gli ID possono contenere underscore, per esempio `15_1`.

## Benchmark dei requisiti hardware

Il benchmark completo su sette viste ha misurato un picco di circa 10,85 GiB di
VRAM e 11,2 GiB di RAM. Per lavorare con margine sono consigliati una GPU NVIDIA
CUDA con almeno 16 GiB di VRAM e 32 GiB di RAM; 16 GiB di RAM rappresentano il
minimo misurato. Tempi e memoria dipendono da numero di viste, risoluzione e
stack software.

Per ripetere la misura senza cache:

```powershell
python tools/benchmark_inference.py --require-cuda
```

`tools/analyze_requirements.py` esegue invece un controllo non distruttivo di
hardware, cache e spazio disponibile.

## Struttura del repository

```text
GraphReconstruction/
├── run.py                    # ricostruzione
├── evaluate.py               # metriche e confronto con gli standard
├── visualize.py              # rendering del grafo finale
├── json_adapter_to_qgis.py   # export GeoJSON
├── graph_reconstruction/
│   ├── config.py             # configurazione, soglie e vincoli
│   ├── pipeline/             # fasi della ricostruzione
│   ├── shared/               # detection e funzioni comuni
│   └── visualization/        # bird view e simboli
├── evaluation/               # confronto con i grafi di riferimento
├── tools/                    # diagnostica, benchmark e packaging
├── models/                   # checkpoint YOLO
├── assets/                   # primitive grafiche
└── output/                   # destinazione predefinita dei risultati
```

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

La pipeline scrive sotto `WORK_DIR`:

```text
output/
├── detection_cache/               # cache YOLO e decoder
├── multiview/
│   └── node<id>/
│       ├── fused_graph.json        # risultato principale
│       ├── scene_nodes.json        # tracciabilità della ReID
│       ├── vggt_geometry.json      # geometria aggregata
│       ├── vggt/*.npz              # point map per vista
│       └── fused_graph_plot.png    # creato da visualize.py
├── standard_comparison/            # report di evaluate.py
└── debug/                           # dump facoltativi
```

Per diagnosticare un nodo, abilitare temporaneamente:

```dotenv
DEBUG_OUTPUT_ENABLED=true
```

Rieseguire poi soltanto l'ID interessato. I dump vengono salvati sotto
`DEBUG_OUTPUT_DIR/node<id>/`; al termine riportare l'opzione a `false`. La guida
ai singoli strumenti è in [`tools/debug/README.md`](tools/debug/README.md).

## Valutazione e standard

`evaluate.py` confronta detection, istanze e archi con i dati di riferimento e
aggiorna `WORK_DIR/standard_comparison/`. `visualize.py` produce il render del
grafo e il report HTML del miglior standard compatibile. Un'esecuzione limitata
a pochi ID aggiorna i report per quel sottoinsieme; generare i report definitivi
solo dopo avere stabilito il batch finale.

## QGIS

Per esportare tutti i grafi disponibili:

```powershell
python json_adapter_to_qgis.py
```

Per un singolo grafo, preferendo la geometria VGGT quando disponibile:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --geometry-source auto
```

Opzioni, layout e attributi sono descritti in [`docs/QGIS.md`](docs/QGIS.md).

## Documentazione

- [`docs/PIPELINE.md`](docs/PIPELINE.md): flusso algoritmico;
- [`docs/QGIS.md`](docs/QGIS.md): export e attributi GIS;
- [`tools/debug/README.md`](tools/debug/README.md): diagnosi dei risultati;
- `docs/technical_manual/main.pdf`: manuale operativo e tecnico completo.
