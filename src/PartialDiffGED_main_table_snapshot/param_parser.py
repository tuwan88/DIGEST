"""Minimal CLI parser for the standalone dense DiffGED runner."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run minimal dense PartialDiffGED.")

    def add_arg(*flags, old_flags=(), **kwargs):
        parser.add_argument(*flags, **kwargs)
        if old_flags:
            alias_kwargs = {
                key: kwargs[key]
                for key in ("dest", "action", "type", "choices", "nargs", "const", "metavar")
                if key in kwargs
            }
            alias_kwargs["default"] = argparse.SUPPRESS
            alias_kwargs["help"] = argparse.SUPPRESS
            parser.add_argument(*old_flags, **alias_kwargs)

    parser.add_argument("--dataset", type=str, default="AIDS", help="Dataset name.")
    parser.add_argument("--abs-path", type=str, default="../", help="Project root that contains json_data/.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional data root that contains json_data/ or json_data_min_15k_testpairs/. Defaults to --abs-path.")
    parser.add_argument("--fixed-pair-root", type=str, default=None, help="Optional root containing fixed-pair benchmark folders with benchmark_<split>_pairs.jsonl and App-BMao text graphs.")
    parser.add_argument("--model-name", type=str, default="PartialDiffMatch", help="Checkpoint name suffix.")
    parser.add_argument("--model-path", type=str, default="model_save/", help="Checkpoint directory.")
    parser.add_argument("--result-path", type=str, default="result/", help="Evaluation result directory.")

    parser.add_argument("--model-train", type=int, default=1, help="1 for train, 0 for test.")
    parser.add_argument("--model-epoch-start", type=int, default=0, help="Checkpoint epoch to load before running.")
    parser.add_argument("--model-epoch-end", type=int, default=0, help="Final epoch index.")
    parser.add_argument("--save-every-epochs", type=int, default=10, help="Checkpoint save interval during training.")
    parser.add_argument(
        "--validation-enable",
        action="store_true",
        help="Run validation on the val split during training using reverse decoding and test_k=1.",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Validation interval during training. Values <= 0 disable periodic validation.",
    )

    parser.add_argument("--denoise-network", choices=["lightgt_dense", "diffged_sparse"], default="lightgt_dense")
    parser.add_argument("--diffusion-mechanism", choices=["partialdiff", "diffged"], default="partialdiff")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 64, 32, 32])
    parser.add_argument("--gt-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--dense-topk-enable", action="store_true")
    parser.add_argument("--dense-topk-start-layer", type=int, default=2)
    parser.add_argument("--dense-topk-row", type=int, default=16)
    parser.add_argument("--dense-topk-col", type=int, default=16)
    parser.add_argument("--dense-topk-score-source", choices=["qk_mean", "qk_max", "z_l2"], default="qk_mean")
    parser.add_argument("--dense-topk-force-current-matching", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument(
        "--test-nodes",
        type=int,
        default=0,
        help="Evaluation dense-node-pair budget per batch when --test-batch-size-bucketing is enabled. Values > 0 replace --test-batch-size with max(1, floor(test_nodes / (max_n1 * max_n2))) for each size bucket; values <= 0 keep --test-batch-size.",
    )
    parser.add_argument(
        "--test-batch-size-bucketing",
        action="store_true",
        help="During evaluation, sort graph pairs by similar dense tensor size before batching to reduce padding waste.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--load-workers", type=int, default=0, help="Parallel workers for startup data loading. 0 means auto.")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--num-testing-graphs", type=int, default=100)
    parser.add_argument(
        "--num-delta-graphs",
        type=int,
        default=0,
        help="Number of synthetic delta graph pairs generated for each large graph.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-diffusion_steps", type=int, default=10)
    parser.add_argument(
        "--partialdiff-noise-mode",
        choices=["fixed_count", "bernoulli_drop"],
        default="fixed_count",
        help="PartialDiff training corruption: fixed-count subset (legacy) or independent Bernoulli keep/drop on GT=1 edges only. Both modes never create 0->1 edges.",
    )
    parser.add_argument(
        "--partialdiff-keep-schedule",
        choices=["linear", "alpha_bar"],
        default="linear",
        help="PartialDiff GT-edge survival schedule: legacy linear 1-t/T or the one-way categorical diffusion alpha_bar.",
    )

    parser.add_argument("--topk-approach", choices=["parallel"], default="parallel")
    parser.add_argument("--test-k", type=int, default=100)
    parser.add_argument(
        "--test-k-small",
        type=int,
        default=0,
        help="IMDB-only override for small fixed-pair test cases. Values <= 0 fall back to --test-k.",
    )
    parser.add_argument(
        "--test-k-large",
        type=int,
        default=0,
        help="IMDB-only override for large fixed-pair test cases. Values <= 0 fall back to --test-k.",
    )
    parser.add_argument(
        "--save-test-k-candidates",
        action="store_true",
        help="Save final probability maps and matchings for all test-k candidates of every graph pair.",
    )
    parser.add_argument(
        "--save-raw-pair-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-pair raw JSONL logs with graph ids, GEDs, edge counts, and matchings. Use --no-save-raw-pair-log for faster evaluation-only runs.",
    )
    add_arg(
        "--refine-enable",
        old_flags=("--app-bmao-postprocess-enable",),
        dest="app_bmao_postprocess_enable",
        action="store_true",
        help="Run GED refinement directly after dense inference without requiring a saved candidate .pt file.",
    )
    add_arg(
        "--refine-mode",
        old_flags=("--app-bmao-postprocess-mode",),
        dest="app_bmao_postprocess_mode",
        choices=["best", "all", "beam", "avg"],
        default="best",
        help="Use the best diffusion candidate, all saved candidates, a beam-style top-B candidate subset before taking the per-pair minimum refined GED, or average the saved probability/score matrices into one synthetic candidate before refinement.",
    )
    add_arg(
        "--app-bmao-search-states",
        dest="app_bmao_search_states",
        type=int,
        default=100,
        help="External App-BMao search-state budget passed as -k.",
    )
    add_arg(
        "--refine-anchor-ratio",
        old_flags=("--app-bmao-anchor-ratio",),
        dest="app_bmao_anchor_ratio",
        type=float,
        default=0.6,
        help="Keep the highest-probability fraction of matched edges as hard anchors before refinement.",
    )
    add_arg(
        "--app-bmao-workers",
        dest="app_bmao_workers",
        type=int,
        default=1,
        help="Parallel worker count for the external App-BMao refinement stage.",
    )
    add_arg(
        "--gpu-search-batch-size",
        old_flags=("--gpu-pair-chunk-size", "--app-bmao-pair-chunk-size"),
        dest="app_bmao_pair_chunk_size",
        type=int,
        default=0,
        metavar="SEARCH_BATCH_SIZE",
        help="Maximum number of graph pairs sent to one internal gpu_refine search batch. Values <= 0 process all pairs at once.",
    )
    add_arg(
        "--app-bmao-overlap-enable",
        action="store_true",
        help="Stream external App-BMao refine in background threads while GPU diffusion continues on later test batches.",
    )
    add_arg(
        "--gpu-overlap-delay-batches",
        old_flags=("--app-bmao-gpu-refine-overlap-delay-batches",),
        dest="app_bmao_gpu_refine_overlap_delay_batches",
        type=int,
        default=0,
        help="For --refine-backend gpu_refine, build refine contexts on CPU in a background thread and delay GPU refine by this many diffusion batches.",
    )
    add_arg(
        "--app-bmao-stdin-enable",
        action="store_true",
        help="Send db/query/anchor payloads directly to the stdin-enabled external App-BMao binary instead of writing temp files.",
    )
    add_arg(
        "--refine-candidate-budget",
        old_flags=("--app-bmao-candidate-budget",),
        dest="app_bmao_candidate_budget",
        type=int,
        default=4,
        help="Candidate budget used when --refine-mode beam. Values <= 0 fall back to all candidates. Ignored by best/all/avg.",
    )
    add_arg(
        "--app-bmao-ged-bin",
        type=str,
        default=str(REPO_ROOT / "third_party/App-BMao-tlimited_main_table/ged"),
        help="Path to the external anchored App-BMao ged binary.",
    )
    add_arg(
        "--dfs-bmao-ged-bin",
        type=str,
        default=str(REPO_ROOT / "third_party/DFS-BMao/ged"),
        help="Path to the external anchored DFS-BMao ged binary.",
    )
    add_arg(
        "--astar-bmao-ged-bin",
        type=str,
        default=str(REPO_ROOT / "third_party/Astar-Bmao/ged"),
        help="Path to the external anchored Astar-BMao ged binary.",
    )
    add_arg(
        "--external-bmao-lower-bound",
        choices=["LSa", "BMao", "BMa"],
        default="BMao",
        help="Lower-bound method passed to external BMao-family solvers via -l.",
    )
    add_arg(
        "--app-bmao-timeout-seconds",
        type=float,
        default=30.0,
        help="Timeout per external App-BMao subprocess call during integrated postprocess.",
    )
    add_arg(
        "--app-bmao-diffusion-ub-enable",
        action="store_true",
        help="For external BMao-family solvers, pass diffusion candidate GED as the initial upper bound via -t candidate_ged-1 and fall back to the diffusion matching if no strictly better solution is found.",
    )
    add_arg(
        "--app-bmao-disable-incumbent-ub",
        action="store_true",
        help="Disable passing the candidate matching and candidate GED as an incumbent upper bound to external App-BMao.",
    )
    add_arg(
        "--refine-backend",
        old_flags=("--app-bmao-search-backend",),
        dest="app_bmao_search_backend",
        choices=["external_app_bmao", "external_dfs_bmao", "external_astar_bmao", "gpu_refine"],
        default="external_app_bmao",
        help="Choose the refine backend: an external BMao-family subprocess or the internal gpu_refine tensor backend.",
    )
    add_arg(
        "--gpu-assignment-backend",
        old_flags=("--app-bmao-assignment-backend",),
        dest="app_bmao_assignment_backend",
        choices=["auto", "scipy", "torch_linear_assignment", "greedy", "row_top1_unique_n2"],
        default="auto",
        help="Assignment/completion backend used by the internal gpu_refine path.",
    )
    add_arg(
        "--gpu-profile-enable",
        old_flags=("--app-bmao-profile-enable",),
        dest="app_bmao_profile_enable",
        action="store_true",
        help="Collect detailed timing breakdowns for the internal gpu_refine path and save them into the output summary JSON.",
    )
    parser.add_argument(
        "--profile-runtime-enable",
        action="store_true",
        help="Collect per-batch wall-time and CUDA memory snapshots for dense inference and integrated postprocess.",
    )
    add_arg(
        "--gpu-fast-exact-update",
        dest="app_bmao_fast_exact_update",
        action="store_true",
        default=False,
        help="Enable the strict-improvement fast path during gpu_refine exact update.",
    )
    add_arg(
        "--gpu-refine-diagnostics-enable",
        dest="app_bmao_refine_diagnostics_enable",
        action="store_true",
        default=False,
        help="Collect per-pair gpu_refine diagnostic counters.",
    )
    add_arg(
        "--gpu-beam-width",
        old_flags=("--v9-beam-width",),
        dest="v9_beam_width",
        type=int,
        default=4,
        help="Fixed beam width w used by gpu_refine.",
    )
    add_arg(
        "--gpu-branch-width",
        old_flags=("--v9-branch-width",),
        dest="v9_branch_width",
        type=int,
        default=4,
        help="Per-state top-b action count kept before the per-pair top-w beam prune in gpu_refine.",
    )
    add_arg(
        "--gpu-candidate-cap",
        old_flags=("--v9-candidate-cap",),
        dest="v9_candidate_cap",
        type=int,
        default=8,
        help="Per-row candidate column cap q used by gpu_refine before adding the delete action.",
    )
    add_arg(
        "--gpu-max-search-depth",
        old_flags=("--v9-max-search-depth",),
        dest="v9_max_search_depth",
        type=int,
        default=0,
        help="Maximum non-anchor node-order expansion steps in gpu_refine; <=0 searches the complete available order.",
    )
    add_arg(
        "--gpu-lb-type",
        old_flags=("--v9-lb-type",),
        dest="v9_lb_type",
        choices=["row_col_min"],
        default="row_col_min",
        help="Cheap lower-bound type used by gpu_refine.",
    )
    add_arg(
        "--gpu-rerank-pool",
        old_flags=("--v9-rerank-pool",),
        dest="v9_rerank_pool",
        type=int,
        default=32,
        help="Per-pair rerank pool size used by gpu_refine before exact GED beam reselection.",
    )
    add_arg(
        "--gpu-lb-tiebreak-weight",
        old_flags=("--v9-lb-tiebreak-weight",),
        dest="v9_lb_tiebreak_weight",
        type=float,
        default=0.01,
        help="Small lower-bound tie-break weight added after exact rerank in gpu_refine.",
    )

    parser.add_argument("--experiment", choices=["test"], default="test")
    parser.add_argument("--testset", choices=["test", "val", "small", "large"], default="test")
    parser.add_argument(
        "--inference-pair-mode",
        choices=["dataset_pairs", "k_graph_cross_product"],
        default="dataset_pairs",
        help="Inference input mode: use the original dataset pair list or expand K selected graphs into pair combinations.",
    )
    parser.add_argument(
        "--inference-graph-count",
        type=int,
        default=0,
        help="Number of graphs K used when --inference-pair-mode k_graph_cross_product.",
    )
    parser.add_argument(
        "--inference-graph-split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Source split for selecting K graphs in k-graph cross-product inference mode.",
    )
    parser.add_argument(
        "--inference-graph-offset",
        type=int,
        default=0,
        help="Start offset inside the selected split when choosing K graphs for cross-product inference.",
    )
    parser.add_argument(
        "--inference-pair-symmetry",
        choices=["unique", "ordered"],
        default="unique",
        help="Whether K-graph cross-product inference keeps only unique unordered pairs or all ordered pairs.",
    )
    parser.add_argument(
        "--eval-precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Evaluation autocast precision. Applied only when --model-train 0 on CUDA.",
    )
    parser.add_argument(
        "--constrained-greedy-mode",
        choices=[
            "global_n3",
            "conf_row_greedy_n2",
            "row_top1_unique_n2",
            "col_top1_unique_n2",
            "alternating_row_col_top1_n2",
            "alternating_col_row_top1_n2",
        ],
        default="global_n3",
    )

    parser.add_argument("--match-label-weight", type=float, default=1.0)
    parser.add_argument("--match-degree-weight", type=float, default=1.0)
    parser.add_argument("--match-similarity-weight", type=float, default=1.0)
    parser.add_argument("--match-cost-scale", type=float, default=1.0)
    parser.add_argument(
        "--reverse-decode-mode",
        choices=["constrained", "blockwise_autoregressive", "none"],
        default="constrained",
        help="Whether each PartialDiff reverse step projects scores to a legal partial matching before the next step.",
    )
    parser.add_argument(
        "--score-calibration-mode",
        choices=["calibrated", "raw"],
        default="calibrated",
        help="Use logit(prob)-base_cost for matching scores, or raw denoiser logits without base-cost correction.",
    )
    parser.add_argument(
        "--reverse-row-top1-repair-mode",
        choices=["final_step", "every_step"],
        default="final_step",
        help="When using row_top1_unique_n2 reverse decoding, repair only the final reverse step or every reverse step.",
    )
    parser.add_argument(
        "--diagnostic-rowtop1-global-overlap",
        action="store_true",
        help="When using row_top1_unique_n2, also decode global_n3 at each reverse step and report overlap with unrepaired Row-top1.",
    )
    parser.add_argument("--renoise-mode", choices=["stochastic", "topk", "none"], default="stochastic")
    parser.add_argument("--blockwise-renoise-mode", choices=["fixed_count", "bernoulli_drop"], default="fixed_count", help="For blockwise_autoregressive reverse only: fixed-count legacy re-noise or independent one-way Bernoulli drop on newly decoded edges. Locked edges remain 1.")
    parser.add_argument("--renoise-temperature", type=float, default=1.0)

    return parser.parse_args()
