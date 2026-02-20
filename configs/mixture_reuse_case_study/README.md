# Mixture Reuse Case Study

This directory contains configs for to walk through a mixture reuse case study from the paper (Section 5.1's mixture reuse method, c=3, seed=0). The workflow builds a mixture over 5 updates, 4 of which freeze the existing ratios and recomputes only on the affected domains. Note that removing a domain (update 4) does not require recomputation, so we omit configs for that update.

- `fit/` — fit configs for reproducing the regression procedure and proposed mix from each update (using swarm data hosted on HuggingFace)
- `generate/` — generation configs used to produce the swarms.

## Walkthrough

Each update follows the same pattern: generate a swarm, train proxy models to get metrics, then fit to propose the next mixture. The fit step is fully reproducible from HuggingFace data. The training step is internal to AI2, requiring S3 data paths.

Each fit config expects `ratios.csv` and `metrics.csv` in the directory you run `olmix` from. Set your repo root once:

```bash
export OLMIX_REPO=/path/to/olmix
```

---

### Update 1 — Add stack-edu

Freeze the DCLM topic mix from a prior swarm; vary the stack-edu language breakdown.

**Generate** (`generate/1_add_stackedu.yaml`): samples 64 swarm variants with DCLM weights frozen and stack-edu weights free.

```bash
olmix generate \
  --config $OLMIX_REPO/configs/mixture_reuse_case_study/generate/1_add_stackedu.yaml \
  --base $OLMIX_REPO/configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update1_variants/
```

This produces one `LaunchConfig` YAML per variant (e.g. `mixture-reuse-swarm-add-stackedu-a1b2c3d4-0000.yaml`). Each file contains the full training config for one proxy run, including the sampled `mix` field showing the domain weights. The S3 data paths are internal to AI2 so the configs can't be launched externally, but you can inspect the `mix` field in each file to see what mixture that proxy model should be trained on.

The actual swarm results from this step (the ratios and metrics CSVs from training the proxy models) are already hosted on HuggingFace, which is what the fit step below uses. If you can't run the proxy models yourself, you can skip straight to **Fit** — this is a simulated walkthrough.

**Fit**:
```bash
mkdir -p /tmp/olmix_reuse/update1 && cd /tmp/olmix_reuse/update1
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update1_add_stack_edu_seed0/ratios.csv"
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update1_add_stack_edu_seed0/metrics.csv"
olmix fit --config $OLMIX_REPO/configs/mixture_reuse_case_study/fit/1_add_stackedu.yaml --output-dir output/
```

---

### Update 2 — Add more sources

Freeze the entire Update 1 mixture as `existing`; vary six new sources (algebraicstack, arxiv, finemath-3plus, pes2o, s2pdfv2, wikipedia).

**Generate** (`generate/2_add_more_sources.yaml`): samples 16 swarm variants with `existing` frozen and new sources free.

```bash
olmix generate \
  --config $OLMIX_REPO/configs/mixture_reuse_case_study/generate/2_add_more_sources.yaml \
  --base $OLMIX_REPO/configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update2_variants/
```

Inspect the `mix` field in any output YAML to see the per-domain weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
mkdir -p /tmp/olmix_reuse/update2 && cd /tmp/olmix_reuse/update2
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update2_add_more_sources_seed0/ratios.csv"
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update2_add_more_sources_seed0/metrics.csv"
olmix fit --config $OLMIX_REPO/configs/mixture_reuse_case_study/fit/2_add_more_sources.yaml --output-dir output/
```

---

### Update 3 — Revise PDFs

Freeze the Update 2 mixture; vary s2pdfv1 (a revised version of the PDF source) against it.

**Generate** (`generate/3_revise_pdfs.yaml`): samples 16 variants with `existing` frozen and s2pdfv1 free.

```bash
olmix generate \
  --config $OLMIX_REPO/configs/mixture_reuse_case_study/generate/3_revise_pdfs.yaml \
  --base $OLMIX_REPO/configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update3_variants/
```

Inspect the `mix` field in any output YAML to see the per-domain weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
mkdir -p /tmp/olmix_reuse/update3 && cd /tmp/olmix_reuse/update3
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update3_revise_pdfs_seed0/ratios.csv"
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update3_revise_pdfs_seed0/metrics.csv"
olmix fit --config $OLMIX_REPO/configs/mixture_reuse_case_study/fit/3_revise_pdfs.yaml --output-dir output/
```

---

### Update 4 — Remove algebraicstack

Algebraicstack is dropped from the mixture. No recomputation needed (mixture reuse), so no configs for this update.

---

### Update 5 — Partition PDFs by topic

Freeze the Update 3/4 mixture; vary s2pdfv1 split into 21 topic partitions.

**Generate** (`generate/5_partition_pdfs.yaml`): samples 16 variants with `existing` frozen and s2pdfv1 topic partitions free.

```bash
olmix generate \
  --config $OLMIX_REPO/configs/mixture_reuse_case_study/generate/5_partition_pdfs.yaml \
  --base $OLMIX_REPO/configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update5_variants/
```

Inspect the `mix` field in any output YAML to see the per-domain weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
mkdir -p /tmp/olmix_reuse/update5 && cd /tmp/olmix_reuse/update5
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update5_partition_pdfs_seed0/ratios.csv"
wget -q "https://huggingface.co/datasets/allenai/olmix/resolve/main/mixture_reuse/real_world/full_reuse/update5_partition_pdfs_seed0/metrics.csv"
olmix fit --config $OLMIX_REPO/configs/mixture_reuse_case_study/fit/5_partition_pdfs.yaml --output-dir output/
```
