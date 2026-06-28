# Confronto con gli standard di riferimento

Questa cartella contiene la logica integrata da `evaluate.py` per confrontare
ogni `fused_graph_qgis_combined.geojson` con i dodici grafi manuali.

Il confronto usa un massimo sottografo comune indotto e label-aware. Lo
shurikode 53 è opzionale: viene contratto e non influenza MCS, GED proxy o
ranking. Il ranking massimizza i nodi MCS, poi gli archi MCS e infine minimizza
la GED proxy.

Esecuzione nel flusso normale:

```powershell
python evaluate.py
python evaluate.py 2 7 9
```

Esecuzione autonoma:

```powershell
python -m evaluation.standard_comparison.confronta_pipeline_con_standard --nodes 2 7 9
python -m evaluation.standard_comparison.visualizza_best_match
python -m evaluation.standard_comparison.visualizza_standard_riferimento
python -m evaluation.standard_comparison.analizza_converti_grafi_manuali
```

Output aggregati: `output/standard_comparison/`.
Ogni nodo riceve inoltre `output/multiview/nodeX/standard_similarity.json`.
`visualize.py` genera anche `output/standard_comparison/report_completo.html`,
con grafo ricostruito e best match affiancati in un unico documento.
