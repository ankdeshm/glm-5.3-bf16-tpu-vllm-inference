# Cloud TPU vLLM Benchmark Report — `v7x-16-spot-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T145414Z`
- **TPU Slice / Config**: `v7x-16-spot-2x2x4-32cores` (`glm53-bf16-tp4-dp8-ep32-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3-BF16`
- **Sanity Check**: `3.946s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **42.23** | **46.64** | 42.23 | 2313.0 | 2329.2 | 21.44 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **158.41** | **42.95** | 39.69 | 2015.4 | 2999.7 | 23.29 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **288.54** | **41.22** | 36.11 | 3532.1 | 4052.0 | 24.27 | 16/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | **448.13** | **31.12** | 28.1 | 3577.1 | 4824.1 | 32.18 | 32/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | **752.25** | **26.95** | 23.94 | 4852.0 | 7660.5 | 37.19 | 64/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | **1147.33** | **20.7** | 18.18 | 6810.3 | 12391.5 | 48.67 | 128/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 1 | **42.52** | **46.62** | 42.53 | 2141.7 | 2291.5 | 21.45 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 4 | **159.24** | **42.87** | 39.83 | 1839.6 | 2300.8 | 23.33 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 8 | **306.92** | **41.94** | 38.4 | 2271.9 | 2888.2 | 23.86 | 16/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 16 | **485.27** | **32.94** | 30.36 | 2673.0 | 3416.9 | 30.36 | 32/0 |
| context_sweep | 2048 | 1024 | 0% | 4 | **144.71** | **38.56** | 36.74 | 1371.4 | 1767.3 | 26.32 | 8/0 |
| context_sweep | 8192 | 1024 | 0% | 4 | **143.74** | **41.59** | 35.99 | 3860.3 | 4748.0 | 24.06 | 8/0 |
| context_sweep | 15000 | 1024 | 0% | 4 | **126.36** | **38.69** | 31.8 | 5832.0 | 8947.5 | 25.93 | 8/0 |
