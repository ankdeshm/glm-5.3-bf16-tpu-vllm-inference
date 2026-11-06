# Atlas Cloud TPU Benchmark Report — `v7x-16-spot-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T201157Z`
- **TPU Slice / Config**: `v7x-16-spot-2x2x4-32cores` (`glm53-bf16-tp4-dp8-ep32-mtp3-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3-BF16`
- **Sanity Check**: `2.684s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **51.69** | **59.54** | 52.22 | 2399.4 | 2424.0 | 17.02 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **140.85** | **51.73** | 46.11 | 2244.3 | 3349.6 | 20.68 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **135.97** | **49.42** | 43.26 | 2540.9 | 2948.5 | 22.96 | 7/9 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | ERR | ERR | ERR | - | - | - | 0/32 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | ERR | ERR | ERR | - | - | - | 0/64 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | ERR | ERR | ERR | - | - | - | 0/128 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 1 | ERR | ERR | ERR | - | - | - | 0/8 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 4 | ERR | ERR | ERR | - | - | - | 0/8 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 8 | ERR | ERR | ERR | - | - | - | 0/16 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 16 | ERR | ERR | ERR | - | - | - | 0/32 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 32 | ERR | ERR | ERR | - | - | - | 0/64 |
| atlas_80pct_prefix_cache | 4096 | 1024 | 80% | 64 | ERR | ERR | ERR | - | - | - | 0/128 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 1 | ERR | ERR | ERR | - | - | - | 0/8 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 4 | ERR | ERR | ERR | - | - | - | 0/8 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 8 | ERR | ERR | ERR | - | - | - | 0/16 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 16 | ERR | ERR | ERR | - | - | - | 0/32 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 32 | ERR | ERR | ERR | - | - | - | 0/64 |
| atlas_95pct_prefix_cache | 4096 | 1024 | 95% | 64 | ERR | ERR | ERR | - | - | - | 0/128 |
| context_sweep | 2048 | 1024 | 0% | 4 | ERR | ERR | ERR | - | - | - | 0/8 |
| context_sweep | 8192 | 1024 | 0% | 4 | ERR | ERR | ERR | - | - | - | 0/8 |
| context_sweep | 15000 | 1024 | 0% | 4 | ERR | ERR | ERR | - | - | - | 0/8 |
