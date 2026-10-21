# Pipeline di ricostruzione

Questo documento descrive la pipeline attiva dopo il refactoring strutturale.
Il punto di ingresso è `run.py`; l’implementazione vive interamente in
`graph_reconstruction/pipeline/`.

## Flusso generale

```text
immagini
  ↓
detection YOLO + decoding Shurikode + cache
  ↓
grafi osservati per singola vista
  ↓
ReID multi-vista delle istanze
  ↓
geometria e suggerimenti 3D VGGT
  ↓
costruzione diretta degli archi
  ↓
connettività / spanning tree
  ↓
annotazione dei diametri
  ↓
fused_graph.json + scene_nodes.json
```

## 1. Discovery delle immagini

`shared/detection.py::discover_nodes` cerca in `IMAGES_DIR` file con pattern
`node<id>_<frame>.jpg`. L’identificatore può contenere underscore, per esempio
`node17_2_0042.jpg` appartiene al nodo `17_2`.

`run.py` può elaborare tutti i gruppi scoperti oppure soltanto gli ID passati
da riga di comando.

## 2. Detection e decoding

`pipeline/observed_graphs.py` richiama
`shared/detection.py::detect_and_decode_cached` per ogni immagine.

Su cache miss:

1. l’immagine viene migliorata con CLAHE;
2. YOLO propone le bounding box;
3. il decoder Shurikode assegna un codice a ogni crop;
4. vengono aggiunti nome, valenza, centro normalizzato e confidence;
5. il risultato viene salvato in `WORK_DIR/detection_cache/`.

La firma della cache include versione e soglie YOLO. Una variazione dei
parametri invalida automaticamente i record incompatibili.

## 3. Grafi osservati

Ogni vista diventa un `ObservedGraph`: le detection accettate sono nodi locali
con riferimento all’immagine e alla bounding box. Questi grafi conservano
l’evidenza per immagine e costituiscono l’input della ReID.

## 4. Artefatti VGGT

Se `VGGT_EDGE_ENABLED=true`, `vggt_runtime.py` verifica la presenza degli NPZ
in `output/multiview/node<id>/vggt/`. Se mancano e la generazione è consentita,
avvia `vggt_export_cli.py` con `VGGT_EMBEDDED_PYTHON`.

Gli NPZ sono costosi da rigenerare e vengono mantenuti come cache persistente.
Per i nodi elencati in `VGGT_SKIP_NODES`, il passaggio viene saltato perché le
viste non hanno overlap sufficiente.

## 5. Re-identification multi-vista

`node_reid.py` raggruppa le osservazioni che rappresentano la stessa istanza
fisica. Usa:

- incompatibilità tra detection co-visibili;
- codice componente e confidence;
- firme strutturali dei vicinati;
- distanza tra nuvole o centroidi VGGT;
- vincoli anchor/terminal definiti in `branch_fragments.py`.

Il risultato è una lista di scene node con codice fuso, immagini di supporto,
confidence e riferimenti alle osservazioni originali. Un passaggio successivo
unisce singleton spurii quando evidenza 3D e struttura lo consentono.

Con `DEBUG_OUTPUT_ENABLED=true` vengono salvati:

```text
DEBUG_OUTPUT_DIR/node<id>/scene_nodes_reid_raw.json
DEBUG_OUTPUT_DIR/node<id>/reid.json
DEBUG_OUTPUT_DIR/node<id>/reid_geometry.json
```

## 6. Suggerimenti 3D per gli archi

`vggt_3d_edges.py` associa le osservazioni alle point map VGGT, stima geometria
e distanze tra scene node e produce suggerimenti per le coppie candidate.

`vggt_geometry.json` contiene sia la geometria utile a QGIS e bird view sia la
cache delle decisioni VGGT. Il file storico `vggt_edge_debug.json` viene ancora
letto per compatibilità.

## 7. Costruzione del grafo

`direct_graph_builder.py` combina:

- evidenza degli observed graph;
- prossimità e mutual-neighbor 3D;
- valenze fisiche;
- co-occorrenze e coppie proibite definite in `config.py`.

Il builder produce il grafo candidato e una lista di bridge disponibili per il
passaggio di connettività.

## 8. Refinement e connettività

`graph_refinement.py::force_spanning_tree` connette le componenti quando il
grafo candidato non è ancora connesso, privilegiando i bridge con evidenza
migliore. Lo stesso modulo serializza nodi e archi nel formato finale.

## 9. Annotazione dei diametri

`diameter_annotation.py` rileva le osservazioni di diametro, le associa agli
archi e propaga il valore rispettando i componenti che interrompono
legittimamente la continuità del diametro.

Con diagnostica abilitata salva:

```text
DEBUG_OUTPUT_DIR/node<id>/diameters.json
```

## 10. Artefatti finali

Per ogni nodo la pipeline scrive in `WORK_DIR/multiview/node<id>/`:

| File | Ruolo |
| --- | --- |
| `fused_graph.json` | Grafo finale, input di evaluate, visualize e QGIS. |
| `scene_nodes.json` | Istanze fisiche fuse e osservazioni di supporto. |
| `vggt_geometry.json` | Geometria 3D e cache VGGT. |
| `vggt/*.npz` | Point map e predizioni per immagine. |
| `bird_view.png` | Proiezione planare della geometria 3D. |

`visualize.py` aggiunge `fused_graph_plot.png`; `evaluate.py` può aggiungere
metriche e similarità standard.

## Moduli della pipeline

| Modulo | Responsabilità |
| --- | --- |
| `orchestrator.py` | Sequenza, logging e output. |
| `observed_graphs.py` | Rappresentazione per vista. |
| `node_reid.py` | Fusione delle istanze multi-vista. |
| `branch_fragments.py` | Classificazione anchor e terminal. |
| `vggt_runtime.py` | Lifecycle degli artefatti VGGT. |
| `vggt_export_cli.py` | Inferenza VGGT nel runtime embedded. |
| `vggt_3d_edges.py` | Geometria, vote log e suggerimenti archi. |
| `direct_graph_builder.py` | Costruzione diretta del grafo. |
| `graph_refinement.py` | Connettività e serializzazione. |
| `diameter_annotation.py` | Diametri sugli archi. |

## Cache e riproducibilità

- La cache detection viene invalidata dalla propria firma.
- Gli artefatti VGGT vengono riutilizzati se tutte le viste attese esistono.
- La cache geometrica VGGT è valida soltanto quando gli ID dei scene node e la
  versione interna coincidono.
- I percorsi configurabili sono risolti rispetto alla root del progetto.

Per forzare una detection nuova su una singola immagine si può chiamare
`detect_and_decode_cached(..., force=True)` da codice. Non eliminare in massa
tutti i JSON di `output`: `fused_graph.json`, `scene_nodes.json` e la geometria
VGGT sono artefatti funzionali, non semplice debug.

## Diagnostica

La diagnostica automatica è disabilitata per default per mantenere gli output
compatti. Impostare `DEBUG_OUTPUT_ENABLED=true`, rieseguire il nodo e consultare
[`../tools/debug/README.md`](../tools/debug/README.md) per scegliere lo script
adatto.
