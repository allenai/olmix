# Mixture Reuse Case Study

This directory contains configs for to walk through a mixture reuse case study from the paper (Section 5.1's mixture reuse method, c=3, seed=0). The workflow begins with an initial DCLM-only swarm, then builds a mixture over 5 updates, 4 of which freeze the existing ratios and recompute only on the affected domains. Note that removing a domain (update 4) does not require recomputation, so we omit configs for that update.

- `fit/` — fit configs for reproducing the regression procedure and proposed mix from each update (using swarm data hosted on HuggingFace)
- `generate/` — generation configs used to produce the swarms.

## Walkthrough

Each update follows the same pattern: generate a swarm, train proxy models to get metrics, then fit to propose the next mixture. The fit step is fully reproducible from HuggingFace data. The training step is internal to AI2, requiring S3 data paths.

Clone the repo and the HuggingFace data once, then run all commands from the repo root:

```bash
git clone https://github.com/allenai/olmix
cd olmix
git clone https://huggingface.co/datasets/allenai/olmix hf-data
```

---

### Stage 0 — Initial DCLM swarm

Mix over 24 DCLM topics.

**Generate** (`generate/0_dclm.yaml`): samples 128 swarm variants over DCLM topics.

```bash
olmix generate \
  --config configs/mixture_reuse_case_study/generate/0_dclm.yaml \
  --base configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/stage0_variants/
```

This produces one `LaunchConfig` YAML per variant (e.g. `mixture-reuse-swarm-dclm-a1b2c3d4-0000.yaml`). Each file contains the full training config for one proxy run, including the sampled `mix` field showing the domain weights. The S3 data paths are internal to AI2 so the configs can't be launched externally, but you can inspect the `mix` field in each file to see what mixture that proxy model should be trained on.

The actual swarm results from this step (the ratios and metrics CSVs from training the proxy models) are already hosted on HuggingFace, which is what the fit step below uses. If you can't run the proxy models yourself, you can skip straight to **Fit** — this is a simulated walkthrough.

**Fit**:
```bash
cp hf-data/dclm_swarm/{ratios,metrics}.csv .
olmix fit --config configs/mixture_reuse_case_study/fit/0_dclm.yaml --output-dir output/
```

The fit step proposes a mixture. These weights are then embedded as the `weight:` field on each DCLM topic in `generate/1_add_stackedu.yaml` in the next step, freezing the DCLM mix so that Update 1's swarm only varies the new stack-edu programming languages.

---

### Update 1 — Add stack-edu

Reuse the DCLM topic mix; recompute over DCLM and the stack-edu programming languages.

**Generate** (`generate/1_add_stackedu.yaml`): samples 64 swarm variants with DCLM weights frozen and stack-edu weights free.

```bash
olmix generate \
  --config configs/mixture_reuse_case_study/generate/1_add_stackedu.yaml \
  --base configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update1_variants/
```

Inspect the `mix` field in any output YAML to see the weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
cp hf-data/mixture_reuse/real_world/full_reuse/update1_add_stack_edu_seed0/{ratios,metrics}.csv .
olmix fit --config configs/mixture_reuse_case_study/fit/1_add_stackedu.yaml --output-dir output/
```

The proposed mixture (DCLM + stack-edu ratios) is carried forward as the frozen `existing` source in `generate/2_add_more_sources.yaml` in the next step.

---

### Update 2 — Add more sources

Reuse the entire Update 1 mixture as `existing`; recompute six new sources (algebraicstack, arxiv, finemath-3plus, pes2o, PDFs, wikipedia).

**Generate** (`generate/2_add_more_sources.yaml`): samples 16 swarm variants with `existing` frozen and new sources free.

```bash
olmix generate \
  --config configs/mixture_reuse_case_study/generate/2_add_more_sources.yaml \
  --base configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update2_variants/
```

Inspect the `mix` field in any output YAML to see the weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
cp hf-data/mixture_reuse/real_world/full_reuse/update2_add_more_sources_seed0/{ratios,metrics}.csv .
olmix fit --config configs/mixture_reuse_case_study/fit/2_add_more_sources.yaml --output-dir output/
```

The proposed mixture is carried forward as the frozen `existing` source in `generate/3_revise_pdfs.yaml`.

---

### Update 3 — Revise PDFs

Reuse the Update 2 mixture; vary s2pdfv1 (a revised version of the PDF source) against it.

**Generate** (`generate/3_revise_pdfs.yaml`): samples 16 variants with `existing` relative ratios frozen and s2pdfv1 free.

```bash
olmix generate \
  --config configs/mixture_reuse_case_study/generate/3_revise_pdfs.yaml \
  --base configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update3_variants/
```

Inspect the `mix` field in any output YAML to see the weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
cp hf-data/mixture_reuse/real_world/full_reuse/update3_revise_pdfs_seed0/{ratios,metrics}.csv .
olmix fit --config configs/mixture_reuse_case_study/fit/3_revise_pdfs.yaml --output-dir output/
```

The proposed mixture is carried forward as the frozen `existing` source in `generate/5_partition_pdfs.yaml`, after algebraicstack is dropped in Update 4.

---

### Update 4 — Remove algebraicstack

Algebraicstack is dropped from the mixture. No recomputation is needed, so there are no configs for this update.

---

### Update 5 — Partition PDFs by topic

Reuse the Update 3/4 mixture; recompute the PDFs split into 21 topic domains.

**Generate** (`generate/5_partition_pdfs.yaml`): samples 64 variants with `existing` frozen and PDF topic domains free.

```bash
olmix generate \
  --config configs/mixture_reuse_case_study/generate/5_partition_pdfs.yaml \
  --base configs/mixture_reuse_case_study/launch/base.yaml \
  --output output/update5_variants/
```

Inspect the `mix` field in any output YAML to see the weights sampled for that variant. The swarm results are on HuggingFace, so skip to **Fit** if you can't run the proxy models.

**Fit**:
```bash
cp hf-data/mixture_reuse/real_world/full_reuse/update5_partition_pdfs_seed0/{ratios,metrics}.csv .
olmix fit --config configs/mixture_reuse_case_study/fit/5_partition_pdfs.yaml --output-dir output/
```

The proposed mix, once expanded, is the mix over the final 64 domains that is evaluated downstream in our paper.
