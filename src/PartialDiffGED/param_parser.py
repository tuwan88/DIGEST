"""Getting params from the command line."""

import argparse


def comma_separated_floats(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    value = str(value).strip()
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run PartialDiffGED.")
    
    parser.add_argument('--topk-approach', choices=['parallel'], default='parallel', help="Choose the top-k mapping generation approach. PartialDiffGED uses the partial-diffusion parallel decoder.")
    
    parser.add_argument('--test-k', type=int, default=100,help='Set k for inference.')
    
    parser.add_argument('--k-range', type=list, default=[1,10,20,30,40,50,60,70,80,90,100],help='range of k for top-k approach analysis.')
    
    parser.add_argument('--experiment', choices=['test', 'topk_analysis', 'diversity_analysis', 'artifact_local_search'],default='test', help="Choose an experiment: test, topk_analysis, diversity_analysis, or artifact_local_search.")
    
    parser.add_argument('--testset', choices=['test', 'small', 'large'],default='test', help="Choose a testing graph set: test, small, or large.")

    parser.add_argument('--diffusion-steps', type=int, default=1000)

    parser.add_argument('--inference-diffusion_steps', type=int, default=10)
    parser.add_argument("--inference-forward-only-prefix-steps",
                        type=int,
                        default=0,
                        help="Run only reverse_step_forward for the first N inference diffusion steps, then resume normal forward+decode+renoise.")
    parser.add_argument("--log-step-ged-curve",
                        action="store_true",
                        help="Log per-diffusion-step GED curve (best-of-k per pair) during inference.")
    parser.add_argument("--timing-breakdown",
                        action="store_true",
                        help="Print stage-level timing breakdowns during both training and evaluation.")
    parser.add_argument("--module-timing-breakdown",
                        action="store_true",
                        help="Profile per-module forward/backward timing during training and save a visualization. This adds synchronization overhead, so use it only for small profiling runs.")
    parser.add_argument("--module-timing-topk",
                        type=int,
                        default=20,
                        help="How many modules to show in the saved module timing chart.")
    parser.add_argument("--disable-tqdm",
                        action="store_true",
                        help="Disable tqdm progress bars and print plain logs instead.")
    parser.add_argument("--tqdm-position",
                        type=int,
                        default=0,
                        help="Terminal row used by tqdm when multiple jobs run concurrently.")
    parser.add_argument("--match-label-weight",
                        type=float,
                        default=1.0,
                        help="Weight for label mismatch in the matching decoder score.")
    parser.add_argument("--match-degree-weight",
                        type=float,
                        default=1.0,
                        help="Weight for node degree difference in the matching decoder score.")
    parser.add_argument("--match-similarity-weight",
                        type=float,
                        default=1.0,
                        help="Weight for node feature similarity in the matching decoder score.")
    parser.add_argument("--match-cost-scale",
                        type=float,
                        default=1.0,
                        help="Global scale applied to the decoder cost adjustment.")
    parser.add_argument("--renoise-mode",
                        choices=["stochastic", "topk"],
                        default="stochastic",
                        help="How to sample the next partial matching from a clean matching.")
    parser.add_argument("--renoise-temperature",
                        type=float,
                        default=1.0,
                        help="Temperature used by stochastic re-noising.")
    parser.add_argument("--inference-decoder",
                        choices=["constrained_greedy"],
                        default="constrained_greedy",
                        help="Inference-only hard decoder (slim build keeps constrained_greedy only).")
    parser.add_argument("--constrained-greedy-mode",
                        choices=["global_n3", "conf_row_greedy_n2", "row_top1_unique_n2"],
                        default="global_n3",
                        help="Implementation used when --inference-decoder constrained_greedy.")

    parser.add_argument("--hidden-dim",
                        type=int,
                        nargs="+",
                        default=[64,64,32,32],
	                help="List of hidden dimensions, e.g. --hidden-dim 64 64 32 32")
    parser.add_argument("--denoise-network",
                        choices=[
                            "lightgt_dense",
                            "lightgt_disjoint",
                        ],
                        default="lightgt_dense",
                        help="Choose denoising backbone (slim build keeps lightgt_dense/lightgt_disjoint only).")
    parser.add_argument("--gt-heads",
                        type=int,
                        default=4,
                        help="Number of attention heads used by the lightgt denoiser option.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="Dropout used in lightgt-style residual branches.")
    parser.add_argument("--enable-topk-pruning",
                        action="store_true",
                        help="Enable optional per-layer candidate pruning inside lightgt/lightgt_notime cross layers.")
    parser.add_argument("--topk-ratios",
                        type=comma_separated_floats,
                        default=[0.9, 0.8, 0.7, 0.7],
                        help="Comma-separated per-layer candidate ratios for top-k pruning, e.g. 0.9,0.8,0.7,0.7")
    parser.add_argument("--topk-min",
                        type=int,
                        default=16,
                        help="Minimum per-row/per-column candidate count kept by adaptive top-k pruning.")
    parser.add_argument("--topk-max",
                        type=int,
                        default=50,
                        help="Maximum per-row/per-column candidate count kept by adaptive top-k pruning.")
    parser.add_argument("--topk-anchor-bias",
                        type=float,
                        default=2.0,
                        help="Extra score bias added to current M_t=1 pairs during lightgt top-k pruning.")
    parser.add_argument("--topk-score-source",
                        choices=["base_cost", "noise"],
                        default="base_cost",
                        help="Score source used to rank candidate pairs for lightgt top-k pruning.")
    parser.add_argument("--log-candidate-recall",
                        action="store_true",
                        help="Log per-layer candidate recall and active-edge ratios for lightgt top-k pruning during training.")
    parser.add_argument("--dense-topk-enable",
                        action="store_true",
                        help="Enable dense cross-layer top-k candidate refinement (for lightgt_dense).")
    parser.add_argument("--dense-topk-start-layer",
                        type=int,
                        default=2,
                        help="Layer index where dense top-k refinement starts (inclusive).")
    parser.add_argument("--dense-topk-row",
                        type=int,
                        default=16,
                        help="Per-row top-k used by dense top-k refinement.")
    parser.add_argument("--dense-topk-col",
                        type=int,
                        default=16,
                        help="Per-column top-k used by dense top-k refinement.")
    parser.add_argument("--dense-topk-score-source",
                        choices=["qk_mean", "qk_max"],
                        default="qk_mean",
                        help="Score source used to build dense top-k candidate mask.")
    parser.add_argument("--dense-topk-force-current-matching",
                        type=int,
                        default=1,
                        help="Whether to force keep current matching edges (noise>0.5) in dense top-k (1/0).")
    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--max-train-batches",
                        type=int,
                        default=0,
                        help="If >0, stop each training epoch early after this many batches. Useful for timing/profiling on a small subset.")
    parser.add_argument("--test-batch-size",
                        type=int,
                        default=1,
                        help="Number of graph pairs per evaluation batch. Default is 1.")
    parser.add_argument("--max-test-batches",
                        type=int,
                        default=0,
                        help="If >0, only evaluate the first N test batches (quick profiling).")
    parser.add_argument("--max-test-pairs",
                        type=int,
                        default=0,
                        help="If >0, only evaluate the first N test graph pairs globally.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=0,
                        help="Number of dataloader workers per process.")
    parser.add_argument("--dist-backend",
                        type=str,
                        default="nccl",
                        help="Distributed backend used when launched with torchrun.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")


    parser.add_argument("--abs-path",
                        type=str,
                        default="../",
                        help="the absolute path")

    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save evaluation results. If left as the default during a fresh training run, a new timestamped subdirectory is created automatically.")
    parser.add_argument("--save-pair-results",
                        action="store_true",
                        help="Save per-pair test results including predicted GED and decoded matchings.")
    parser.add_argument("--save-matching-artifacts",
                        action="store_true",
                        help="During evaluation, save per-pair reverse-step probability maps and final greedy-decoded matchings as torch shard files.")
    parser.add_argument("--artifact-path",
                        type=str,
                        default="",
                        help="Artifact shard file, directory, or glob used by experiment=artifact_local_search.")
    parser.add_argument("--local-search-release-count",
                        type=int,
                        default=4,
                        help="How many low-confidence matched rows to release during artifact_local_search.")
    parser.add_argument("--local-search-extra-cols",
                        type=int,
                        default=1,
                        help="How many currently-unmatched columns to add to the released column pool.")
    parser.add_argument("--local-search-probability-step",
                        type=int,
                        default=-1,
                        help="Which reverse-step probability map to use for confidence scoring. -1 means the last step.")
    parser.add_argument("--local-search-max-pairs",
                        type=int,
                        default=0,
                        help="Optional cap on how many artifact pairs to analyze. 0 means all pairs.")
    parser.add_argument("--local-search-output-prefix",
                        type=str,
                        default="",
                        help="Output prefix for artifact_local_search JSONL/JSON files.")
    parser.add_argument("--local-search-save-all-cases",
                        action="store_true",
                        help="Save every enumerated local assignment and its GED in the per-pair JSONL output.")
    parser.add_argument("--local-search-topk-cases",
                        type=int,
                        default=10,
                        help="How many best local assignments to keep per pair in the JSONL output.")
    parser.add_argument("--two-swap-local-search",
                        action="store_true",
                        help="Enable post-decode best-improvement 2-swap local search on the selected matching.")
    parser.add_argument("--two-swap-max-iter",
                        type=int,
                        default=50,
                        help="Maximum iterations for 2-swap local search.")
    parser.add_argument("--two-swap-eps",
                        type=float,
                        default=1e-9,
                        help="Minimum GED improvement required to accept a swap.")
    parser.add_argument("--two-swap-chunk-size",
                        type=int,
                        default=0,
                        help="Optional chunk size over swap candidates. 0 means evaluate all swaps at once.")
    parser.add_argument("--lowprob-permute-m",
                        type=int,
                        default=0,
                        help="Post-process decoded matching by selecting the m lowest-probability matched edges and trying all column permutations among them. 0 disables this pass.")
    parser.add_argument("--lowprob-permute-max-cases",
                        type=int,
                        default=0,
                        help="Optional cap on enumerated permutation cases per pair for low-probability permutation post-process. 0 means no cap.")

    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save trained checkpoints. If left as the default during a fresh training run, a new timestamped subdirectory is created automatically.")
    parser.add_argument("--save-every-epochs",
                        type=int,
                        default=10,
                        help="If >0, save an intermediate checkpoint every N epochs during training.")

    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=0,
                        help="The number of epochs the final saved model has been trained.")

    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")

    parser.add_argument("--model-name",
                        type=str,
                        default='PartialDiffMatch',
                        help="model name")


    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")

    

    return parser.parse_args()
