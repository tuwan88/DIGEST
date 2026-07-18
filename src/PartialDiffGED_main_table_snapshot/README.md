This directory is a slimmed-down copy of `PartialDiffGED_dense_diffusion_v9opt_external`.

It keeps only three execution modes:
- diffusion-only dense inference
- diffusion + `gpu_refine`
- diffusion + `external_app_bmao`

Retained refine backends:
- `external_app_bmao`
- `gpu_refine`

Core refine arguments:
- `--app-bmao-postprocess-enable`
- `--app-bmao-search-backend {external_app_bmao,gpu_refine}`
- `--app-bmao-postprocess-mode`
- `--app-bmao-search-states`
- `--app-bmao-anchor-ratio`
- `--app-bmao-pair-chunk-size`
- `--v9-beam-width`
- `--v9-branch-width`
- `--v9-candidate-cap`
- `--v9-rerank-pool`
- `--v9-lb-tiebreak-weight`

Notes:
- `external_app_bmao` still invokes the anchored App-BMao binary through the trainer subprocess path.
- `gpu_refine` is the only retained internal GPU refine implementation.
- Old refine variants, auxiliary scripts, and extra backend files are intentionally omitted here.
