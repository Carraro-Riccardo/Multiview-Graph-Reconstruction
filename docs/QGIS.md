# Esportazione QGIS

`json_adapter_to_qgis.py` converte `fused_graph.json` in una FeatureCollection
GeoJSON combinata con componenti, archi e poligono master.

## Batch

Senza input esplicito vengono cercati tutti i grafi sotto
`WORK_DIR/multiview/`:

```powershell
python json_adapter_to_qgis.py
```

Per cambiare la root:

```powershell
python json_adapter_to_qgis.py --batch-root D:/risultati/multiview
```

## Singolo grafo

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json
```

La directory di output predefinita è quella del grafo. È possibile separarla:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --output-dir output/qgis/node2
```

Il file prodotto segue il pattern:

```text
<stem>_qgis_combined.geojson
```

## Sorgente della geometria

`--geometry-source` accetta:

- `fake`: layout sintetico, comportamento predefinito;
- `auto`: usa VGGT quando `vggt_geometry.json` è disponibile, altrimenti
  ripiega sul layout sintetico;
- `vggt`: richiede la geometria VGGT e termina con errore se manca.

Esempio consigliato per ispezione spaziale:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --geometry-source auto
```

Le vecchie cache `vggt_edge_debug.json` restano supportate.

## Layout sintetici

Quando non viene usata la geometria VGGT, `--layout` controlla la disposizione:

- `cad`: usa porte e orientamenti dei simboli Shurikode;
- `circle`: dispone i nodi su una circonferenza;
- `zero`: colloca tutti i nodi sull’origine, utile soltanto per test.

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --layout cad
```

## Coordinate e CRS

Le coordinate di riferimento sono configurabili con:

```text
--origin-x
--origin-y
--radius
--crs-epsg
```

Il CRS dichiarato di default è EPSG:7795. Le coordinate sintetiche non
diventano automaticamente misure reali: origine, raggio e CRS devono essere
coerenti con il progetto GIS in cui il file verrà importato.

## Routing degli archi

`--edge-routing` accetta:

- `orthogonal`: polilinee ad angolo retto;
- `straight`: segmento diretto tra gli estremi.

Il routing ortogonale è il default ed è generalmente più leggibile con layout
CAD.

## Attributi

L’export normale espone attributi orientati all’uso GIS. Per includere anche i
campi interni della pipeline con prefisso `src_`:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --include-technical-fields
```

Il master supporta ID, data e orientamento:

```powershell
python json_adapter_to_qgis.py `
  output/multiview/node2/fused_graph.json `
  --master-id nodo_2_master `
  --master-date 2026-06-28 `
  --master-orientation 90
```

## Packaging

Dopo aver generato gli export combinati per i nodi:

```powershell
python tools/qgis/package_qgis_geojson.py --overwrite
```

Il comando cerca i file combinati sotto `output/multiview/node*/` e produce:

```text
output/qgis_geojson_layers.zip
```

Opzioni disponibili:

```powershell
python tools/qgis/package_qgis_geojson.py --help
```

## Import in QGIS

1. Aprire **Layer → Aggiungi layer → Aggiungi layer vettoriale**.
2. Selezionare il file `*_qgis_combined.geojson`.
3. Filtrare o stilizzare le feature in base al tipo geometrico e agli
   attributi del layer.
4. Verificare che il CRS del progetto sia coerente con `--crs-epsg`.

Il GeoJSON è combinato intenzionalmente: nodi, archi e master condividono un
singolo artefatto distribuibile, ma possono essere filtrati separatamente in
QGIS.
