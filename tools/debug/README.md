# Strumenti di debug

Questa directory contiene diagnostica manuale. Nessuno di questi script viene
eseguito automaticamente da `run.py`: leggono gli input, le cache e i risultati
già prodotti, quindi generano report sotto `DEBUG_OUTPUT_DIR` o stampano
un’analisi nel terminale.

Eseguire sempre i comandi dalla root del progetto:

```powershell
python tools/debug/<script>.py [argomenti]
```

## Preparazione consigliata

Per avere tutti i dump interni ReID e geometrici, impostare in `.env`:

```dotenv
DEBUG_OUTPUT_ENABLED=true
DEBUG_OUTPUT_DIR=./output/debug
```

Poi rieseguire soltanto il nodo da diagnosticare:

```powershell
python run.py 7
```

Gli script non richiedono tutti gli stessi dati. La tabella seguente indica i
prerequisiti principali.

| Script | Scopo | Output principale | Richiede dump pipeline |
| --- | --- | --- | --- |
| `analyze_reid_false_positives.py` | Spiega cluster ReID in eccesso | JSON + terminale | consigliato |
| `diagnose_node.py` | Diagnosi completa over-segmentation | terminale | no |
| `debug_crops.py` | Crop raggruppati per scene node | PNG | no |
| `debug_graph_projection.py` | Proietta nodi e archi sulle immagini | PNG + JSON | no |
| `plot_reid_geometry.py` | Cluster 3D separati per codice | PNG | sì |
| `visualize_pointcloud_bboxes.py` | Point cloud e centroidi fusi | PNG + finestra | no |
| `visualize_reid_3d.py` | Scatter 3D multi-nodo | PNG + finestra | no |
| `visualize_vggt_cluster_projections.py` | Cluster VGGT proiettati per vista | PNG + JSON | no |

## `analyze_reid_false_positives.py`

Confronta i scene node predetti con le istanze ground truth mediante IoU e
Hungarian matching. Per ogni falso positivo cerca il cluster candidato con cui
avrebbe potuto essere unito e riporta evidenza condivisa, immagini di supporto,
affinità e motivi di mancato merge.

```powershell
# Tutti i nodi presenti negli output
python tools/debug/analyze_reid_false_positives.py

# Nodi specifici
python tools/debug/analyze_reid_false_positives.py 4 7 9
```

Legge:

- `output/multiview/node<id>/scene_nodes.json`;
- `output/detection_cache/*.json`;
- label in `LABELS_EXT_DIR`;
- se disponibili, `DEBUG_OUTPUT_DIR/node<id>/scene_nodes_reid_raw.json` e
  `reid.json`.

Produce:

```text
DEBUG_OUTPUT_DIR/node<id>/reid_false_positive_analysis.json
```

Senza i dump opzionali funziona sui scene node finali, ma non può spiegare
tutte le decisioni interne della ReID.

## `diagnose_node.py`

Analisi testuale focalizzata sull’over-segmentation. Mostra scene node non
associati al ground truth, copertura delle istanze tra immagini e viste in cui
il numero di detection è inferiore o superiore alle istanze attese.

```powershell
python tools/debug/diagnose_node.py 7
```

Senza argomento usa il nodo 9. Non scrive file: il report viene stampato nel
terminale. Richiede annotazioni del grafo, label estese, cache detection e
`scene_nodes.json`.

## `debug_crops.py`

Crea una tavola di crop per ogni scene node. Ogni riga permette di confrontare
le detection provenienti dalle varie immagini e, quando disponibile, il match
con il ground truth.

```powershell
python tools/debug/debug_crops.py 3
python tools/debug/debug_crops.py 2 3 9
```

Produce un file per nodo:

```text
DEBUG_OUTPUT_DIR/node<id>/crops.png
```

È utile per riconoscere rapidamente cluster che uniscono componenti diversi o
cluster duplicati della stessa istanza fisica.

## `debug_graph_projection.py`

Sovrappone il grafo finale a ogni immagine originale. Le bounding box sono
colorate per scene node e gli archi sono classificati in base alla coerenza
della proiezione:

- verde: entrambi gli estremi sono visibili e vicini;
- arancione: entrambi visibili ma geometricamente lontani;
- grigio: almeno un estremo non è visibile nella vista.

```powershell
python tools/debug/debug_graph_projection.py 3
python tools/debug/debug_graph_projection.py 2 7 9
```

Produce:

```text
DEBUG_OUTPUT_DIR/node<id>/projections/proj_<image>.png
DEBUG_OUTPUT_DIR/node<id>/projections/consistency_report.json
```

Il JSON aggrega per arco il numero di viste supportate, violate o non
osservabili.

## `plot_reid_geometry.py`

Visualizza la geometria usata dalla ReID separatamente per codice componente.
Ogni punto rappresenta una detection; colore e label mostrano l’assegnazione al
scene node.

```powershell
# Tutti i codici del nodo 7
python tools/debug/plot_reid_geometry.py 7

# Soltanto i codici 15 e 95
python tools/debug/plot_reid_geometry.py 7 15 95
```

Richiede:

```text
DEBUG_OUTPUT_DIR/node<id>/reid_geometry.json
output/multiview/node<id>/scene_nodes.json
```

Il primo file viene prodotto solo con `DEBUG_OUTPUT_ENABLED=true`. Output:

```text
DEBUG_OUTPUT_DIR/node<id>/reid_geometry/code_<codice>.png
```

## `visualize_reid_3d.py`

Mostra uno scatter 3D con più nodi affiancati. Distingue cluster associati a un
ground truth, falsi positivi e istanze mancanti. Ricostruisce i dati necessari
dalle cache e dagli artefatti VGGT.

```powershell
python tools/debug/visualize_reid_3d.py 7
python tools/debug/visualize_reid_3d.py 4 7 9
```

Senza argomenti usa i nodi 7 e 9. Produce:

```text
DEBUG_OUTPUT_DIR/reid_3d_scatter.png
```

Lo script apre anche la finestra Matplotlib quando l’ambiente grafico lo
consente.

## `visualize_vggt_cluster_projections.py`

Proietta sulle immagini originali i cluster costruiti usando la geometria
VGGT. Serve a verificare visivamente se le detection assegnate alla stessa
istanza sono coerenti tra viste.

```powershell
python tools/debug/visualize_vggt_cluster_projections.py 7
python tools/debug/visualize_vggt_cluster_projections.py 7 9
```

Produce:

```text
DEBUG_OUTPUT_DIR/node<id>/cluster_projections/<image>_clusters.png
DEBUG_OUTPUT_DIR/node<id>/cluster_projections/cluster_projection_report.json
```

Il report contiene assegnazioni, conteggi, match ground truth e percorso delle
immagini generate.

## `visualize_pointcloud_bboxes.py`

Mostra una vista originale insieme alla point cloud VGGT e ai centroidi fusi
delle istanze ricostruite:

```powershell
python tools/debug/visualize_pointcloud_bboxes.py 7
```

## Pulizia

Gli output diagnostici sono confinati in `DEBUG_OUTPUT_DIR`, quindi possono
essere rimossi senza cancellare `fused_graph.json`, le cache detection o gli
artefatti VGGT. Non usare script generici che eliminano tutti i JSON sotto
`output/`: alcuni JSON sono risultati o cache necessari.
