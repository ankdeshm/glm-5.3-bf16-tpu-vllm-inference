# `zai-org/GLM-5.3-BF16` (`1.51 TB` Unquantized BF16) Serving & Benchmarking on Cloud TPU `v7x`

End-to-end multi-host serving configuration, benchmark harness, and empirical latency/throughput results for the unquantized **[`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16)** (`744B` parameters, **`1,506.7 GB` Pure BF16** across `282` safetensors shards) on a **single TPU `v7x-16` slice (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm 0.32.0` + `vllm-torchtpu` + `TorchTPU 0.2.0` GA**.

---

## 1. Model Overview — [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16) (`282` safetensors shards, `1,506.7 GB` BF16) |
| **Architecture** | `GlmMoeDsaForCausalLM` (`78` layers + `1` MTP layer, `256` routed experts, top-8 routing, DSA Sparse MLA) |
| **Parallelism Layout** | `TP=4, DP=8, EP=32` across `32` TensorCores (`8` routed BF16 experts per TensorCore via EP-filtered Run:AI GCS streamer) |
| **Memory Footprint** | `53.39 GiB/rank` weights (`31.87 GiB/rank` free) -> **`5,193,728` (`5.19M`) FP8 MLA KV-cache tokens** (`649,216`/DP engine) |

---

## 2. Benchmark Results (`v7x-16` Pure BF16 vs `GLM-5.3` Block-FP8 at Identical `TP=4, DP=8, EP=32`)

Because TPU `v7x` holds `3,072 GiB` HBM3e with `118 TB/s` aggregate HBM bandwidth, the entire `1.51 TB` unquantized BF16 checkpoint fits on **one `v7x-16` slice** with `5.19M` KV tokens of headroom — and **outperforms `128×128` block-FP8 by up to `+41.2%` across `c=1..16`** by executing native BF16 MXU matmuls without runtime block-FP8 weight dequantization overhead!

| Workload Suite | Input : Output | Prefix Hit | Concurrency | Pure BF16 Out TPS | vs Block-FP8 | Mean Stream TPS | Mean TTFT (ms) | Mean TPOT (ms) | P90 TPOT (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `1` | **`42.23`** | **`+40.8%`** | **`46.64`** | `2,313.0` | **`21.44`** | `21.47` |
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `4` | **`158.41`** | **`+34.9%`** | **`42.95`** | `2,015.4` | **`23.29`** | `23.62` |
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `8` | **`288.54`** | **`+22.3%`** | **`41.22`** | `3,532.1` | **`24.27`** | `24.89` |
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `16` | **`448.13`** | **`+4.3%`** | **`31.12`** | `3,577.1` | **`32.18`** | `33.42` |
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `32` | `752.25` | `-2.1%` | `26.95` | `4,852.0` | `37.19` | `38.64` |
| `concurrency_0pct_cache` | `4,096 : 1,024` | `0%` | `64` | `1,147.33` | `-8.4%` | `20.70` | `6,810.3` | `48.67` | `52.89` |
| `cached_80pct_prefix` | `4,096 : 1,024` | `80%` | `1` | **`42.52`** | **`+41.2%`** | **`46.62`** | `2,141.7` | **`21.45`** | `21.48` |
| `cached_80pct_prefix` | `4,096 : 1,024` | `80%` | `4` | **`159.24`** | **`+35.3%`** | **`42.87`** | `1,839.6` | **`23.33`** | `23.86` |
| `cached_80pct_prefix` | `4,096 : 1,024` | `80%` | `8` | **`306.92`** | **`+25.7%`** | **`41.94`** | `2,271.9` | **`23.86`** | `24.66` |
| `cached_80pct_prefix` | `4,096 : 1,024` | `80%` | `16` | **`485.27`** | **`+3.6%`** | **`32.94`** | `2,673.0` | **`30.36`** | `30.57` |
| `context_sweep` | `2,048 : 1,024` | `0%` | `4` | **`144.71`** | **`+21.8%`** | **`38.56`** | `1,371.4` | **`26.32`** | `29.79` |
| `context_sweep` | `8,192 : 1,024` | `0%` | `4` | **`143.74`** | **`+33.3%`** | **`41.59`** | `3,860.3` | **`24.06`** | `24.11` |
| `context_sweep` | `15,000 : 1,024` | `0%` | `4` | **`126.36`** | **`+27.8%`** | **`38.69`** | `5,832.0` | **`25.93`** | `27.44` |

- **Official `InferenceX` (`c=8`, `16` prompts, `4096:1024`):** **`1,070.42` total tok/s (`+18.2%` vs FP8)** / **`209.74` output tok/s (`+18.2%`)** | Median ITL **`24.39 ms`** (`P90 ITL 25.21 ms`).

---

## 3. Repository Structure & Reproduction

```text
├── README.md
├── docker/
│   └── Dockerfile                              # vllm 0.32.0 + TorchTPU 0.2.0 GA benchmark image
├── scripts/
│   ├── run_tpu_benchmark_sweep.py              # Streaming OpenAI benchmark + prefix-cache + InferenceX driver
│   └── stage_hf_to_gcs.py                      # Parallel 282-shard Hugging Face -> GCS stager
├── k8s/
│   ├── v7x-16-spot-pass4-glm53-bf16.yaml       # GKE LeaderWorkerSet + Benchmark Job for TPU v7x-16
│   └── v5p-32-spot-glm53-bf16.yaml             # GKE LeaderWorkerSet manifest for TPU v5p-32 (2x4x4)
└── results/                                    # Raw JSON & Markdown benchmark outputs
```

```bash
# 1. Stage zai-org/GLM-5.3-BF16 (1.51 TB, 282 shards) to GCS
python3 scripts/stage_hf_to_gcs.py --repo-id zai-org/GLM-5.3-BF16 --gcs-prefix gs://YOUR_GCS_BUCKET/models/zai-org/GLM-5.3-BF16

# 2. Deploy LeaderWorkerSet & Benchmark Job on TPU v7x-16 (2x2x4)
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py
kubectl apply -f k8s/v7x-16-spot-pass4-glm53-bf16.yaml
```
