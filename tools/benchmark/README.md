# Benchmark e requisiti

- `analyze_requirements.py` controlla ambiente, dati e risorse senza eseguire
  l'inferenza completa;
- `benchmark_inference.py` esegue una prova senza cache e misura tempi, RAM e
  VRAM. Per una misura attendibile deve essere eseguito su una GPU dedicata.

Esempi:

```powershell
python tools/benchmark/analyze_requirements.py
python tools/benchmark/benchmark_inference.py --require-cuda
```
