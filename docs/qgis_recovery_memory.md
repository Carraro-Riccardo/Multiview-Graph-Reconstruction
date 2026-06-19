# QGIS Recovery Memory

This note preserves the recovered QGIS context after the Codex session restore on
2026-06-17.

## Restored Codex Sessions

The old Codex state was found at:

```text
C:\Users\ricca\.codex.backup.20260616-165931
```

Before restoring anything, the current state was backed up to:

```text
C:\Users\ricca\.codex.pre-restore.20260617-154154
```

The recovered session index now includes the relevant GraphReconstruction
threads again in:

```text
C:\Users\ricca\.codex\session_index.jsonl
```

Most relevant QGIS threads:

- `019e7379-b6cc-7820-8842-b5d8fe57e42b`: `Adatta fused_graph per QGIS`
- `019eb11f-8f31-7a40-b705-5632f819dc2d`: `Aggiungi layer master e campi`
- `019d919c-1c23-78e0-8b1d-fcb816c90473`: `Associa SVG ai componenti`

If the Codex sidebar does not show the restored threads immediately, restart
Codex Desktop. The live app process may keep its thread list in memory until
restart.

## QGIS Export Logic

Main adapter:

```text
json_adapter_to_qgis.py
```

The adapter converts pipeline `fused_graph.json` files into QGIS-friendly
GeoJSON layers. It supports both single-file mode and batch mode.

Single file:

```powershell
.\python312\python.exe json_adapter_to_qgis.py output\multiview\node5\fused_graph.json
```

Batch mode scans all node folders under `output/multiview`:

```powershell
.\python312\python.exe json_adapter_to_qgis.py
```

Important defaults:

- Batch root: `output/multiview`
- CRS metadata: EPSG `7795`
- Layout: `cad`
- Edge routing: `orthogonal`
- Geometry source default in the CLI help is `fake`, while the conversion code
  also supports `auto` and `vggt`.

Generated layer files per node use the input node folder as output directory.
The adapter writes separate node, edge, master, and combined GeoJSON outputs.

## Geometry Strategy

The original fused graph is topological: it stores components and connections,
not real-world coordinates. The adapter therefore creates QGIS geometries using
one of these strategies:

- `fake`: synthetic layout around an origin, useful for inspection.
- `cad`: uses the Shurikode CAD-style port layout from
  `visualize_shurikode_symbols.py`.
- `vggt` / `auto`: reads `vggt_edge_debug.json`, projects scene centroids into a
  2D plane, and adds VGGT attributes such as centroid and axis components.

Recovered design note from the QGIS thread: VGGT point maps alone are not enough
to convert directly to component geometry. The adapter needs detection bounding
boxes or already-saved scene geometry to know which 3D points belong to which
component. The current pipeline writes useful geometry artifacts such as:

```text
output/multiview/node*/vggt_edge_debug.json
output/multiview/node*/reid_geometry.json
```

## Master Layer

The `Aggiungi layer master e campi` thread added a master node layer. The master
feature is a polygon centered at the graph origin with practical QGIS fields:

- `id`
- `tipo_layer`
- `nome`
- `node_id`
- `etichetta`
- `data`
- `orientamento`

The CLI exposes:

```powershell
--master-id
--master-date
--master-orientation
```

In batch mode, `master_id` defaults per node, using the node folder name.

## Packaging QGIS Outputs

The recovered June 10 thread added:

```text
debug/package_qgis_geojson.py
```

Purpose: collect all per-node QGIS GeoJSON files into one zip archive while
skipping combined layers.

Typical command:

```powershell
.\python312\python.exe debug\package_qgis_geojson.py --overwrite
```

Default output:

```text
output/qgis_geojson_layers.zip
```

The zip uses folders named like `nodo5`, `nodo10`, etc.

## Practical Restart Point

To regenerate QGIS exports for all available fused graphs:

```powershell
.\python312\python.exe json_adapter_to_qgis.py
```

To package the per-layer GeoJSON outputs:

```powershell
.\python312\python.exe debug\package_qgis_geojson.py --overwrite
```

If the goal is visual inspection in QGIS, prefer `cad` or `auto` geometry when
the supporting VGGT debug files exist. If the goal is a stable simple export,
use the default synthetic geometry.
