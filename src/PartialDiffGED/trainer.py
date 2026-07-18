import sys
import time
import os
import re
import itertools
import glob
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from math import exp
from scipy.stats import spearmanr, kendalltau

from models import DiffMatch, DiffEmbedMatcher
from models_lightgt import (
    LightGTDiffMatch,
    DenseLightGTDiffMatch,
    DisjointLightGTDiffMatch,
)
from loss_fn import mapping_loss
from diffusion_schedulers import CategoricalDiffusion,InferenceSchedule
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.utils import dense_to_sparse,to_undirected,sort_edge_index,coalesce,to_dense_adj,remove_self_loops,to_dense_batch,group_argsort,to_networkx
import torch_geometric as pyg
from torch_geometric.nn.pool import global_add_pool,global_mean_pool
import networkx as nx
import operator
import json


class GraphPairDataset(Dataset):
    """Build graph-pair samples lazily from pair specifications."""

    def __init__(self, trainer, pair_specs):
        self.trainer = trainer
        self.pair_specs = pair_specs

    def __len__(self):
        return len(self.pair_specs)

    def __getitem__(self, idx):
        return self.trainer.pack_graph_pair(self.pair_specs[idx])


class DistributedEvalSampler(Sampler):
    """Shard evaluation samples across ranks without padding/duplication."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        dataset_len = len(self.dataset)
        if self.rank >= dataset_len:
            return 0
        return (dataset_len - 1 - self.rank) // self.num_replicas + 1


class ModuleTimingProfiler:
    """Profile per-module timings without charging earlier async CUDA work to the first hook."""

    def __init__(self, trainer, named_modules):
        self.trainer = trainer
        self.named_modules = list(named_modules)
        self.handles = []
        self.forward_totals = defaultdict(float)
        self.forward_counts = defaultdict(int)
        self.backward_totals = defaultdict(float)
        self.backward_counts = defaultdict(int)
        self.forward_starts = defaultdict(list)
        self.backward_starts = defaultdict(list)
        self.forward_pairs = defaultdict(list)
        self.backward_pairs = defaultdict(list)
        self.use_cuda_events = bool(self.trainer.use_gpu)
        self.enabled = True
        self._register_hooks()

    def _sync(self):
        self.trainer.sync_device()

    def _record_start(self, store, name):
        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            store[name].append(event)
            return
        self._sync()
        store[name].append(time.perf_counter())

    def _record_stop(self, starts, pairs, totals, counts, name):
        if self.use_cuda_events:
            start = starts[name].pop() if starts[name] else None
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if start is not None:
                pairs[name].append((start, end))
                counts[name] += 1
            return
        self._sync()
        start = starts[name].pop() if starts[name] else time.perf_counter()
        totals[name] += time.perf_counter() - start
        counts[name] += 1

    def _materialize_cuda_pairs(self):
        if not self.use_cuda_events:
            return
        self._sync()
        for name, events in self.forward_pairs.items():
            if events:
                self.forward_totals[name] += sum(start.elapsed_time(end) for start, end in events) / 1000.0
                events.clear()
        for name, events in self.backward_pairs.items():
            if events:
                self.backward_totals[name] += sum(start.elapsed_time(end) for start, end in events) / 1000.0
                events.clear()

    def _register_hooks(self):
        for name, module in self.named_modules:
            self.handles.append(module.register_forward_pre_hook(self._make_forward_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self.handles.append(module.register_full_backward_pre_hook(self._make_backward_pre_hook(name)))
            self.handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def _make_forward_pre_hook(self, name):
        def hook(module, inputs):
            if not self.enabled:
                return
            self._record_start(self.forward_starts, name)
        return hook

    def _make_forward_hook(self, name):
        def hook(module, inputs, outputs):
            if not self.enabled:
                return
            self._record_stop(
                self.forward_starts,
                self.forward_pairs,
                self.forward_totals,
                self.forward_counts,
                name,
            )
        return hook

    def _make_backward_pre_hook(self, name):
        def hook(module, grad_outputs):
            if not self.enabled:
                return
            self._record_start(self.backward_starts, name)
        return hook

    def _make_backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            self._record_stop(
                self.backward_starts,
                self.backward_pairs,
                self.backward_totals,
                self.backward_counts,
                name,
            )
        return hook

    def manual_forward_start(self, name):
        if not self.enabled:
            return
        self._record_start(self.forward_starts, name)

    def manual_forward_stop(self, name):
        if not self.enabled:
            return
        self._record_stop(
            self.forward_starts,
            self.forward_pairs,
            self.forward_totals,
            self.forward_counts,
            name,
        )

    def reset(self):
        self.forward_totals = defaultdict(float)
        self.forward_counts = defaultdict(int)
        self.backward_totals = defaultdict(float)
        self.backward_counts = defaultdict(int)
        self.forward_starts = defaultdict(list)
        self.backward_starts = defaultdict(list)
        self.forward_pairs = defaultdict(list)
        self.backward_pairs = defaultdict(list)

    def summarize(self):
        self._materialize_cuda_pairs()
        summary = {}
        all_names = set(self.forward_totals.keys()) | set(self.backward_totals.keys())
        for name in all_names:
            forward_s = float(self.forward_totals.get(name, 0.0))
            backward_s = float(self.backward_totals.get(name, 0.0))
            summary[name] = {
                "forward_s": forward_s,
                "forward_calls": int(self.forward_counts.get(name, 0)),
                "backward_s": backward_s,
                "backward_calls": int(self.backward_counts.get(name, 0)),
                "total_s": forward_s + backward_s,
            }
        return summary

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles = []


class Trainer(object):
    """
    A general model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        if self.args.denoise_network not in {"lightgt_dense", "lightgt_disjoint"}:
            raise ValueError(
                "Slim build supports --denoise-network in {lightgt_dense, lightgt_disjoint}, "
                f"got: {self.args.denoise_network}"
            )
        if self.args.inference_decoder != "constrained_greedy":
            raise ValueError(
                "Slim build supports --inference-decoder constrained_greedy only, "
                f"got: {self.args.inference_decoder}"
            )
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []
        self.timing_totals = defaultdict(float)
        self.timing_counts = defaultdict(int)
        self.memory_stage_max = defaultdict(dict)
        self.candidate_pruning_totals = defaultdict(float)
        self.step_ged_totals = defaultdict(float)
        self.init_stage_times = {}
        self.module_timing_profiler = None
        self._setup_distributed()
        self.use_gpu = torch.cuda.is_available()
        if self.is_main_process:
            print("use_gpu =", self.use_gpu)
        self.device = torch.device(f'cuda:{self.local_rank}') if self.use_gpu else torch.device('cpu')
        self._configure_default_output_dirs()
        # Some launch environments report stdout as non-TTY even when users still
        # expect progress bars in the terminal, so gate tqdm only on rank/flag.
        self.show_progress = self.is_main_process and (not self.args.disable_tqdm)
        init_start = time.perf_counter()
        stage_start = time.perf_counter()
        self.log_stage("Starting load_data")
        self.load_data()
        self.init_stage_times["load_data"] = time.perf_counter() - stage_start
        self.log_stage("Finished load_data")
        stage_start = time.perf_counter()
        self.log_stage("Starting transfer_data_to_torch")
        self.transfer_data_to_torch()
        self.init_stage_times["transfer_data_to_torch"] = time.perf_counter() - stage_start
        self.log_stage("Finished transfer_data_to_torch")
        self.delta_graphs = [None] * len(self.graphs)
        if self.args.experiment == "artifact_local_search":
            self.model = None
            self.training_graphs = None
            self.val_graphs = None
            self.testing_graphs = None
            self.testing_graphs_small = None
            self.testing_graphs_large = None
            self.training_data_loader = None
            self.val_data_loader = None
            self.testing_data_loader = None
            self.testing_data_small_loader = None
            self.testing_data_large_loader = None
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            self.small_test_loader = None
            self.large_test_loader = None
            self.init_stage_times["setup_model"] = 0.0
            self.init_stage_times["gen_delta_graphs"] = 0.0
            self.init_stage_times["init_graph_pairs"] = 0.0
            self.init_stage_times["build_dataloaders"] = 0.0
        else:
            stage_start = time.perf_counter()
            self.log_stage("Starting setup_model")
            self.setup_model()
            self._ensure_module_timing_profiler()
            self.init_stage_times["setup_model"] = time.perf_counter() - stage_start
            self.log_stage("Finished setup_model")
            # generate synthetic graphs for large graphs (if any)
            stage_start = time.perf_counter()
            self.log_stage("Starting gen_delta_graphs")
            self.gen_delta_graphs()
            self.init_stage_times["gen_delta_graphs"] = time.perf_counter() - stage_start
            self.log_stage("Finished gen_delta_graphs")
            stage_start = time.perf_counter()
            self.log_stage("Starting init_graph_pairs")
            self.init_graph_pairs()
            self.init_stage_times["init_graph_pairs"] = time.perf_counter() - stage_start
            self.log_stage("Finished init_graph_pairs")
            stage_start = time.perf_counter()
            self.log_stage("Starting build_dataloaders")
            self._build_dataloaders()
            self.init_stage_times["build_dataloaders"] = time.perf_counter() - stage_start
            self.log_stage("Finished build_dataloaders")
        self.init_stage_times["total"] = time.perf_counter() - init_start
        if self.is_main_process:
            print("\nInit timing breakdown:")
            print("stage\tseconds\tshare_%")
            total_init = max(self.init_stage_times["total"], 1e-12)
            for key in [
                "load_data",
                "transfer_data_to_torch",
                "setup_model",
                "gen_delta_graphs",
                "init_graph_pairs",
                "build_dataloaders",
            ]:
                duration = self.init_stage_times.get(key, 0.0)
                print(f"{key}\t{duration:.5f}\t{100.0 * duration / total_init:.2f}")
            print(f"total\t{total_init:.5f}\t100.00")

    def _setup_distributed(self):
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.distributed:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend=self.args.dist_backend)
        self.is_main_process = self.rank == 0

    def _build_data_loader(self, dataset, batch_size, shuffle, drop_last=False):
        sampler = None
        if self.distributed:
            if shuffle:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
            else:
                sampler = DistributedEvalSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
        )
        return loader, sampler

    def _broadcast_run_id(self, run_id):
        if not self.distributed:
            return run_id
        payload = [run_id if self.is_main_process else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _sanitize_path_token(self, value):
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        return token.strip("._-") or "run"

    def _configure_default_output_dirs(self):
        default_model_path = Path("model_save")
        default_result_path = Path("result")

        raw_model_path = Path(self.args.model_path)
        raw_result_path = Path(self.args.result_path)

        uses_default_model_dir = not raw_model_path.is_absolute() and raw_model_path == default_model_path
        uses_default_result_dir = not raw_result_path.is_absolute() and raw_result_path == default_result_path

        # Only mint a fresh experiment directory when we are starting a new
        # training run from epoch 0 and the user did not override the base path.
        should_create_run_dir = self.args.model_train == 1 and self.args.model_epoch_start == 0
        if not should_create_run_dir:
            return

        run_id = os.environ.get("PARTIALDIFFGED_RUN_ID")
        if not run_id:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            dataset = self._sanitize_path_token(self.args.dataset)
            model_name = self._sanitize_path_token(self.args.model_name)
            run_id = f"{dataset}_{model_name}_{timestamp}"
        run_id = self._broadcast_run_id(run_id)

        if uses_default_model_dir:
            self.args.model_path = str(default_model_path / run_id)
        if uses_default_result_dir:
            self.args.result_path = str(default_result_path / run_id)

        if self.is_main_process and (uses_default_model_dir or uses_default_result_dir):
            print(f"[Output] experiment_id={run_id}", flush=True)
            print(f"[Output] model_path={self._resolve_output_dir(self.args.model_path)}", flush=True)
            print(f"[Output] result_path={self._resolve_output_dir(self.args.result_path)}", flush=True)

    def _build_dataloaders(self):
        self.training_data_loader, self.training_sampler = self._build_data_loader(
            self.training_graphs,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.testing_data_loader, self.testing_sampler = self._build_data_loader(
            self.testing_graphs,
            batch_size=self.args.test_batch_size,
            shuffle=False,
        )
        self.testing_data_small_loader, self.testing_small_sampler = self._build_data_loader(
            self.testing_graphs_small,
            batch_size=self.args.test_batch_size,
            shuffle=False,
        )
        self.testing_data_large_loader, self.testing_large_sampler = self._build_data_loader(
            self.testing_graphs_large,
            batch_size=self.args.test_batch_size,
            shuffle=False,
        )

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _gather_objects(self, obj):
        if not self.distributed:
            return [obj]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _all_reduce_tensor(self, tensor, op=dist.ReduceOp.SUM):
        if not self.distributed:
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def close(self, sync=False):
        if self.module_timing_profiler is not None:
            self.module_timing_profiler.close()
            self.module_timing_profiler = None
        if self.distributed and dist.is_initialized():
            if sync:
                dist.barrier()
            dist.destroy_process_group()

    def barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def log_stage(self, message):
        if self.is_main_process:
            print(f"[Init] {message}", flush=True)

    def sync_device(self):
        if self.use_gpu:
            torch.cuda.synchronize(self.device)

    def start_timer(self, sync=False):
        if sync:
            self.sync_device()
        return time.perf_counter()

    def stop_timer(self, start_time, key=None, sync=False, count=1):
        if sync:
            self.sync_device()
        elapsed = time.perf_counter() - start_time
        if key is not None:
            self.timing_totals[key] += elapsed
            self.timing_counts[key] += count
            self.record_memory_snapshot(key)
        return elapsed

    def reset_timing_stats(self):
        self.timing_totals = defaultdict(float)
        self.timing_counts = defaultdict(int)

    def reset_memory_stats(self):
        self.memory_stage_max = defaultdict(
            lambda: {
                "max_alloc_bytes": 0,
                "max_reserved_bytes": 0,
            }
        )
        if self.use_gpu:
            self.sync_device()
            torch.cuda.reset_peak_memory_stats(self.device)

    def record_memory_snapshot(self, key):
        if (not self.use_gpu) or key is None:
            return
        stats = self.memory_stage_max[key]
        stats["max_alloc_bytes"] = max(
            int(stats.get("max_alloc_bytes", 0)),
            int(torch.cuda.memory_allocated(self.device)),
        )
        stats["max_reserved_bytes"] = max(
            int(stats.get("max_reserved_bytes", 0)),
            int(torch.cuda.memory_reserved(self.device)),
        )

    def aggregate_memory_breakdown(self):
        local_payload = {
            "stage_max": dict(self.memory_stage_max),
            "overall": self.memory_summary(),
        }
        gathered_payloads = self._gather_objects(local_payload)
        aggregated_stage = defaultdict(
            lambda: {
                "max_alloc_bytes": 0,
                "max_reserved_bytes": 0,
            }
        )
        overall = {
            "peak_alloc_bytes": 0,
            "peak_reserved_bytes": 0,
            "final_alloc_bytes": 0,
            "final_reserved_bytes": 0,
        }
        for payload in gathered_payloads:
            for key, item in payload.get("stage_max", {}).items():
                aggregated_stage[key]["max_alloc_bytes"] = max(
                    int(aggregated_stage[key]["max_alloc_bytes"]),
                    int(item.get("max_alloc_bytes", 0)),
                )
                aggregated_stage[key]["max_reserved_bytes"] = max(
                    int(aggregated_stage[key]["max_reserved_bytes"]),
                    int(item.get("max_reserved_bytes", 0)),
                )
            overall_payload = payload.get("overall", {})
            for name in overall:
                overall[name] = max(int(overall[name]), int(overall_payload.get(name, 0)))
        return {
            "stage_max": dict(aggregated_stage),
            "overall": overall,
        }

    def memory_summary(self):
        if not self.use_gpu:
            return {
                "peak_alloc_bytes": 0,
                "peak_reserved_bytes": 0,
                "final_alloc_bytes": 0,
                "final_reserved_bytes": 0,
            }
        self.sync_device()
        return {
            "peak_alloc_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            "final_alloc_bytes": int(torch.cuda.memory_allocated(self.device)),
            "final_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }

    @staticmethod
    def _bytes_to_gb(num_bytes):
        return float(num_bytes) / (1024.0 ** 3)

    def print_memory_breakdown(self, aggregated=None):
        if (not self.is_main_process) or (not self.use_gpu):
            return aggregated
        if aggregated is None:
            aggregated = self.aggregate_memory_breakdown()
        print("\nMemory breakdown (CUDA):")
        print("stage\tmax_alloc_gb\tmax_reserved_gb")
        for key, item in sorted(
            aggregated.get("stage_max", {}).items(),
            key=lambda kv: kv[1].get("max_alloc_bytes", 0),
            reverse=True,
        ):
            print(
                f"{key}\t{self._bytes_to_gb(item.get('max_alloc_bytes', 0)):.3f}\t"
                f"{self._bytes_to_gb(item.get('max_reserved_bytes', 0)):.3f}"
            )
        overall = aggregated.get("overall", {})
        print(
            "overall\t{:.3f}\t{:.3f}".format(
                self._bytes_to_gb(overall.get("peak_alloc_bytes", 0)),
                self._bytes_to_gb(overall.get("peak_reserved_bytes", 0)),
            )
        )
        print(
            "final\t{:.3f}\t{:.3f}".format(
                self._bytes_to_gb(overall.get("final_alloc_bytes", 0)),
                self._bytes_to_gb(overall.get("final_reserved_bytes", 0)),
            )
        )
        return aggregated

    def reset_candidate_pruning_stats(self):
        self.candidate_pruning_totals = defaultdict(float)

    def reset_step_ged_stats(self):
        self.step_ged_totals = defaultdict(float)

    def update_step_ged_stats(self, step_idx, best_ged_values):
        if best_ged_values is None:
            return
        values = best_ged_values.detach().float().view(-1)
        if values.numel() == 0:
            return
        self.step_ged_totals["max_step"] = max(float(self.step_ged_totals.get("max_step", 0.0)), float(step_idx + 1))
        self.step_ged_totals[f"step_{step_idx}_sum"] += float(values.sum().item())
        self.step_ged_totals[f"step_{step_idx}_count"] += float(values.numel())

    def aggregate_step_ged_stats(self):
        gathered = self._gather_objects(dict(self.step_ged_totals))
        aggregated = defaultdict(float)
        for payload in gathered:
            for key, value in payload.items():
                if key == "max_step":
                    aggregated[key] = max(float(aggregated.get(key, 0.0)), float(value))
                else:
                    aggregated[key] += float(value)
        return dict(aggregated)

    def step_ged_curve_payload(self, aggregated=None):
        if aggregated is None:
            aggregated = self.aggregate_step_ged_stats()
        max_step = int(aggregated.get("max_step", 0))
        if max_step <= 0:
            return {}
        steps = []
        for step_idx in range(max_step):
            s = float(aggregated.get(f"step_{step_idx}_sum", 0.0))
            c = float(aggregated.get(f"step_{step_idx}_count", 0.0))
            if c <= 0:
                mean_v = None
            else:
                mean_v = s / c
            steps.append(
                {
                    "step": step_idx + 1,
                    "mean_best_ged": mean_v,
                    "count": int(c),
                }
            )
        return {"num_steps": max_step, "steps": steps}

    def update_candidate_pruning_stats(self, stats):
        if not stats or not stats.get("enabled", False):
            return
        self.candidate_pruning_totals["batches"] += 1.0
        self.candidate_pruning_totals["num_layers"] = float(stats.get("num_layers", 0))
        for layer_idx in range(int(stats.get("num_layers", 0))):
            self.candidate_pruning_totals[f"layer_{layer_idx}_active_edges"] += float(stats["active_edges"][layer_idx])
            self.candidate_pruning_totals[f"layer_{layer_idx}_full_edges"] += float(stats["full_edges"][layer_idx])
            self.candidate_pruning_totals[f"layer_{layer_idx}_recall_hits"] += float(stats["recall_hits"][layer_idx])
            self.candidate_pruning_totals[f"layer_{layer_idx}_recall_total"] += float(stats["recall_total"][layer_idx])

    def aggregate_candidate_pruning_stats(self):
        gathered = self._gather_objects(dict(self.candidate_pruning_totals))
        aggregated = defaultdict(float)
        for payload in gathered:
            for key, value in payload.items():
                aggregated[key] += float(value)
        return dict(aggregated)

    def print_candidate_pruning_stats(self, aggregated=None):
        if not self.is_main_process:
            return
        if aggregated is None:
            aggregated = self.aggregate_candidate_pruning_stats()
        num_layers = int(aggregated.get("num_layers", 0))
        batches = max(float(aggregated.get("batches", 0.0)), 1.0)
        if num_layers <= 0:
            return
        print("\nCandidate pruning summary:")
        print("layer\tactive_edges_avg\tactive_ratio\tcandidate_recall")
        for layer_idx in range(num_layers):
            active_edges = float(aggregated.get(f"layer_{layer_idx}_active_edges", 0.0))
            full_edges = float(aggregated.get(f"layer_{layer_idx}_full_edges", 0.0))
            recall_hits = float(aggregated.get(f"layer_{layer_idx}_recall_hits", 0.0))
            recall_total = float(aggregated.get(f"layer_{layer_idx}_recall_total", 0.0))
            active_edges_avg = active_edges / batches
            active_ratio = active_edges / max(full_edges, 1e-8)
            if recall_total > 0.0:
                candidate_recall = recall_hits / recall_total
                recall_str = f"{candidate_recall:.6f}"
            else:
                recall_str = "n/a"
            print(f"layer_{layer_idx}\t{active_edges_avg:.3f}\t{active_ratio:.6f}\t{recall_str}")

    def candidate_pruning_summary_payload(self, aggregated=None):
        if aggregated is None:
            aggregated = self.aggregate_candidate_pruning_stats()
        num_layers = int(aggregated.get("num_layers", 0))
        batches = max(float(aggregated.get("batches", 0.0)), 1.0)
        if num_layers <= 0:
            return {}
        layers = []
        for layer_idx in range(num_layers):
            active_edges = float(aggregated.get(f"layer_{layer_idx}_active_edges", 0.0))
            full_edges = float(aggregated.get(f"layer_{layer_idx}_full_edges", 0.0))
            recall_hits = float(aggregated.get(f"layer_{layer_idx}_recall_hits", 0.0))
            recall_total = float(aggregated.get(f"layer_{layer_idx}_recall_total", 0.0))
            layers.append(
                {
                    "layer": layer_idx,
                    "active_edges_avg": active_edges / batches,
                    "active_ratio": active_edges / max(full_edges, 1e-8),
                    "candidate_recall": (recall_hits / recall_total) if recall_total > 0.0 else None,
                }
            )
        return {
            "batches": batches,
            "num_layers": num_layers,
            "layers": layers,
        }

    def _module_profile_specs(self, model):
        specs = []
        def add(name, module):
            if module is not None:
                specs.append((name, module))
        if isinstance(model, (LightGTDiffMatch, DenseLightGTDiffMatch)):
            add("pair_to_undirected", getattr(model, "pair_to_undirected", None))
            add("noise_attr_cast", getattr(model, "noise_attr_cast", None))
            add("time_pos_embed", getattr(model, "time_pos_embed", None))
            if getattr(model, "time_embed", None) is not None:
                add("time_embed", model.time_embed)
            add("edge_pos_embed", getattr(model, "edge_pos_embed", None))
            add("edge_embed", getattr(model, "edge_embed", None))
            for idx, module in enumerate(model.conv_layers):
                add(f"conv_layers.{idx}", module)
                add(f"conv_layers.{idx}.nn", getattr(module, "nn", None))
                if hasattr(module, "nn") and len(module.nn) >= 5:
                    add(f"conv_layers.{idx}.nn.0", module.nn[0])
                    add(f"conv_layers.{idx}.nn.2", module.nn[2])
            for idx, module in enumerate(model.gns):
                add(f"gns.{idx}", module)
            for idx, module in enumerate(model.cross_layers):
                add(f"cross_layers.{idx}", module)
                add(f"cross_layers.{idx}.edge_in", getattr(module, "edge_in", None))
                add(f"cross_layers.{idx}.q_proj", getattr(module, "q_proj", None))
                add(f"cross_layers.{idx}.k_proj", getattr(module, "k_proj", None))
                add(f"cross_layers.{idx}.v_proj", getattr(module, "v_proj", None))
                add(f"cross_layers.{idx}.edge_bias", getattr(module, "edge_bias", None))
                add(f"cross_layers.{idx}.time_bias", getattr(module, "time_bias", None))
                add(f"cross_layers.{idx}.edge_value", getattr(module, "edge_value", None))
                add(f"cross_layers.{idx}.edge_gate", getattr(module, "edge_gate", None))
                add(f"cross_layers.{idx}.node_time", getattr(module, "node_time", None))
                add(f"cross_layers.{idx}.node_ffn", getattr(module, "node_ffn", None))
                if hasattr(module, "node_ffn") and len(module.node_ffn) >= 5:
                    add(f"cross_layers.{idx}.node_ffn.0", module.node_ffn[0])
                    add(f"cross_layers.{idx}.node_ffn.1", module.node_ffn[1])
                    add(f"cross_layers.{idx}.node_ffn.4", module.node_ffn[4])
                add(f"cross_layers.{idx}.node_norm", getattr(module, "node_norm", None))
                add(f"cross_layers.{idx}.src_proj", getattr(module, "src_proj", None))
                add(f"cross_layers.{idx}.dst_proj", getattr(module, "dst_proj", None))
                add(f"cross_layers.{idx}.qk_proj", getattr(module, "qk_proj", None))
                add(f"cross_layers.{idx}.attn_proj", getattr(module, "attn_proj", None))
                add(f"cross_layers.{idx}.edge_time", getattr(module, "edge_time", None))
                add(f"cross_layers.{idx}.edge_ffn", getattr(module, "edge_ffn", None))
                if hasattr(module, "edge_ffn") and len(module.edge_ffn) >= 5:
                    add(f"cross_layers.{idx}.edge_ffn.0", module.edge_ffn[0])
                    add(f"cross_layers.{idx}.edge_ffn.1", module.edge_ffn[1])
                    add(f"cross_layers.{idx}.edge_ffn.4", module.edge_ffn[4])
                add(f"cross_layers.{idx}.edge_norm", getattr(module, "edge_norm", None))
                add(f"cross_layers.{idx}.node_out", getattr(module, "node_out", None))
            add("mapMatrix", model.mapMatrix)
            if hasattr(model, "mapMatrix") and len(model.mapMatrix) >= 5:
                add("mapMatrix.0", model.mapMatrix[0])
                add("mapMatrix.2", model.mapMatrix[2])
                add("mapMatrix.4", model.mapMatrix[4])
        else:
            for name, module in model.named_children():
                add(name, module)
        return specs

    def _ensure_module_timing_profiler(self):
        if not self.args.module_timing_breakdown:
            return
        if self.module_timing_profiler is not None:
            return
        specs = [(name, module) for name, module in self._module_profile_specs(self.unwrap_model()) if module is not None]
        self.module_timing_profiler = ModuleTimingProfiler(self, specs)
        model = self.unwrap_model()
        for idx, module in enumerate(getattr(model, "cross_layers", [])):
            if hasattr(module, "configure_manual_timing"):
                module.configure_manual_timing(self.module_timing_profiler, f"cross_layers.{idx}")

    def aggregate_module_timing_breakdown(self):
        if self.module_timing_profiler is None:
            return {}
        local_summary = self.module_timing_profiler.summarize()
        gathered = self._gather_objects(local_summary)
        aggregated = defaultdict(
            lambda: {
                "forward_sum_s": 0.0,
                "forward_max_s": 0.0,
                "forward_calls_sum": 0,
                "backward_sum_s": 0.0,
                "backward_max_s": 0.0,
                "backward_calls_sum": 0,
                "total_sum_s": 0.0,
                "total_max_s": 0.0,
            }
        )
        valid_ranks = max(len(gathered), 1)
        for payload in gathered:
            for name, item in payload.items():
                aggregated[name]["forward_sum_s"] += float(item.get("forward_s", 0.0))
                aggregated[name]["forward_max_s"] = max(aggregated[name]["forward_max_s"], float(item.get("forward_s", 0.0)))
                aggregated[name]["forward_calls_sum"] += int(item.get("forward_calls", 0))
                aggregated[name]["backward_sum_s"] += float(item.get("backward_s", 0.0))
                aggregated[name]["backward_max_s"] = max(aggregated[name]["backward_max_s"], float(item.get("backward_s", 0.0)))
                aggregated[name]["backward_calls_sum"] += int(item.get("backward_calls", 0))
                aggregated[name]["total_sum_s"] += float(item.get("total_s", 0.0))
                aggregated[name]["total_max_s"] = max(aggregated[name]["total_max_s"], float(item.get("total_s", 0.0)))
        for name in aggregated:
            aggregated[name]["forward_avg_s"] = aggregated[name]["forward_sum_s"] / valid_ranks
            aggregated[name]["backward_avg_s"] = aggregated[name]["backward_sum_s"] / valid_ranks
            aggregated[name]["total_avg_s"] = aggregated[name]["total_sum_s"] / valid_ranks
            aggregated[name]["forward_calls_avg"] = aggregated[name]["forward_calls_sum"] / valid_ranks
            aggregated[name]["backward_calls_avg"] = aggregated[name]["backward_calls_sum"] / valid_ranks
        return dict(aggregated)

    def print_module_timing_breakdown(self, total_pairs, aggregated=None):
        if not self.args.module_timing_breakdown or not self.is_main_process:
            return aggregated
        if aggregated is None:
            aggregated = self.aggregate_module_timing_breakdown()
        if not aggregated:
            return aggregated
        total_time = sum(item["total_avg_s"] for item in aggregated.values())
        total_time = max(total_time, 1e-12)
        print("\nTiming breakdown (modules):")
        print("module\tforward_s\tbackward_s\ttotal_s\tshare_%\tforward_ms/call\tbackward_ms/call\tms/pair")
        sorted_items = sorted(aggregated.items(), key=lambda item: item[1]["total_avg_s"], reverse=True)
        for name, item in sorted_items:
            share = 100.0 * item["total_avg_s"] / total_time
            forward_ms = 1000.0 * item["forward_avg_s"] / max(item["forward_calls_avg"], 1.0)
            backward_ms = 1000.0 * item["backward_avg_s"] / max(item["backward_calls_avg"], 1.0)
            ms_per_pair = 1000.0 * item["total_avg_s"] / max(total_pairs, 1)
            print(
                f"{name}\t{item['forward_avg_s']:.5f}\t{item['backward_avg_s']:.5f}\t{item['total_avg_s']:.5f}\t"
                f"{share:.2f}\t{forward_ms:.3f}\t{backward_ms:.3f}\t{ms_per_pair:.3f}"
            )
        return aggregated

    def save_module_timing_artifacts(self, total_pairs, aggregated=None):
        if not self.args.module_timing_breakdown or not self.is_main_process:
            return
        if aggregated is None:
            aggregated = self.aggregate_module_timing_breakdown()
        if not aggregated:
            return
        topk = max(int(self.args.module_timing_topk), 1)
        sorted_items = sorted(aggregated.items(), key=lambda item: item[1]["total_avg_s"], reverse=True)
        selected_items = sorted_items[:topk]
        epoch_value = int(getattr(self, "cur_epoch", self.args.model_epoch_start) + 1)
        output = {
            "epoch": epoch_value,
            "total_pairs": int(total_pairs),
            "modules": [
                {
                    "module": name,
                    **item,
                }
                for name, item in sorted_items
            ],
        }
        json_path = self._result_path(f"module_timing_epoch_{epoch_value}.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        labels = [name for name, _ in selected_items]
        forward_vals = [item["forward_avg_s"] for _, item in selected_items]
        backward_vals = [item["backward_avg_s"] for _, item in selected_items]
        fig_height = max(4.5, 0.42 * len(labels) + 1.8)
        fig, axes = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)

        ax_fwd, ax_bwd = axes
        ax_fwd.barh(labels, forward_vals, color="#4c78a8")
        ax_fwd.invert_yaxis()
        ax_fwd.set_xlabel("Average seconds per rank")
        ax_fwd.set_title("Forward")
        for idx, value in enumerate(forward_vals):
            ax_fwd.text(value + 0.005, idx, f"{value:.3f}s", va="center", fontsize=9)

        ax_bwd.barh(labels, backward_vals, color="#e45756")
        ax_bwd.set_xlabel("Average seconds per rank")
        ax_bwd.set_title("Backward")
        for idx, value in enumerate(backward_vals):
            ax_bwd.text(value + 0.005, idx, f"{value:.3f}s", va="center", fontsize=9)

        fig.suptitle(
            f"Module timing breakdown (epoch {epoch_value}, top {len(labels)})\n"
            f"{self.args.dataset} | {self.args.denoise_network} | {total_pairs} pairs",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        png_path = self._result_path(f"module_timing_epoch_{epoch_value}.png")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[Profiling] Saved module timing JSON to {json_path}", flush=True)
        print(f"[Profiling] Saved module timing chart to {png_path}", flush=True)

    def aggregate_timing_breakdown(self):
        local_payload = {
            "totals": dict(self.timing_totals),
            "counts": dict(self.timing_counts),
        }
        gathered_payloads = self._gather_objects(local_payload)
        aggregated = defaultdict(
            lambda: {
                "sum_s": 0.0,
                "max_s": 0.0,
                "avg_s": 0.0,
                "calls_sum": 0,
                "calls_avg": 0.0,
            }
        )
        valid_ranks = max(len(gathered_payloads), 1)

        for payload in gathered_payloads:
            payload_totals = payload.get("totals", {})
            payload_counts = payload.get("counts", {})
            for key in set(payload_totals.keys()) | set(payload_counts.keys()):
                total_duration = float(payload_totals.get(key, 0.0))
                call_count = int(payload_counts.get(key, 0))
                aggregated[key]["sum_s"] += total_duration
                aggregated[key]["max_s"] = max(aggregated[key]["max_s"], total_duration)
                aggregated[key]["calls_sum"] += call_count

        for key in aggregated:
            aggregated[key]["avg_s"] = aggregated[key]["sum_s"] / valid_ranks
            aggregated[key]["calls_avg"] = aggregated[key]["calls_sum"] / valid_ranks

        return dict(aggregated)

    def print_named_timing_breakdown(self, title, total_units, baseline_key, prefix, unit_label, aggregated=None):
        if not self.args.timing_breakdown or not self.is_main_process:
            return

        if aggregated is None:
            aggregated = self.aggregate_timing_breakdown()
        baseline_total = aggregated.get(baseline_key, {}).get("avg_s", 0.0)
        if baseline_total <= 0:
            return aggregated

        print(f"\nTiming breakdown ({title}):")
        print(
            "stage\tavg_s/rank\tmax_s(rank)\tshare_%\t"
            "avg_ms_per_call\tavg_ms_per_{unit}\tcalls_avg/rank".format(unit=unit_label)
        )
        sorted_items = sorted(
            (
                (key, item)
                for key, item in aggregated.items()
                if key.startswith(prefix) and key != baseline_key
            ),
            key=lambda item: item[1]["avg_s"],
            reverse=True,
        )
        for key, item in sorted_items:
            duration = item["avg_s"]
            calls = item["calls_avg"]
            share = 100.0 * duration / baseline_total
            avg_ms_per_call = 1000.0 * duration / max(calls, 1.0)
            avg_ms_per_unit = 1000.0 * duration / max(total_units, 1)
            print(
                f"{key}\t{duration:.5f}\t{item['max_s']:.5f}\t{share:.2f}\t"
                f"{avg_ms_per_call:.3f}\t{avg_ms_per_unit:.3f}\t{calls:.2f}"
            )
        print(
            f"{baseline_key}\t{baseline_total:.5f}\t{aggregated.get(baseline_key, {}).get('max_s', baseline_total):.5f}\t"
            f"100.00\t-\t{1000.0 * baseline_total / max(total_units, 1):.3f}\t1.00"
        )
        return aggregated

    def print_timing_breakdown(self, total_pairs, aggregated=None):
        aggregated = self.print_named_timing_breakdown(
            "evaluation",
            total_pairs,
            baseline_key="score/total",
            prefix="score/",
            unit_label="pair",
            aggregated=aggregated,
        )
        self.print_named_timing_breakdown(
            "decode",
            total_pairs,
            baseline_key="decode/total",
            prefix="decode/",
            unit_label="pair",
            aggregated=aggregated,
        )
        return aggregated

    def print_training_timing_breakdown(self, total_pairs, aggregated=None):
        return self.print_named_timing_breakdown(
            "training",
            total_pairs,
            baseline_key="train/total",
            prefix="train/",
            unit_label="pair",
            aggregated=aggregated,
        )

    @staticmethod
    def to_python_scalar(value):
        return value.item() if torch.is_tensor(value) else value

    @staticmethod
    def _timing_entry_to_seconds(entry):
        if isinstance(entry, dict):
            return float(entry.get("avg_s", entry.get("sum_s", 0.0)))
        if entry is None:
            return 0.0
        return float(entry)

    @staticmethod
    def normalize_batch_outputs(model_out):
        return model_out if isinstance(model_out, list) else [model_out]

    @staticmethod
    def stack_batch_predictions(batch_outputs, device):
        pred_ged = torch.stack([pair_out[0].detach().to(device=device) for pair_out in batch_outputs], dim=0)
        running_time = torch.tensor([pair_out[2] for pair_out in batch_outputs], device=device, dtype=torch.float32)
        return pred_ged, running_time

    @staticmethod
    def extract_pre_swap_ged(batch_out):
        if len(batch_out) <= 4 or not isinstance(batch_out[4], dict):
            return None
        pre_swap = batch_out[4].get("pre_swap_ged", None)
        if pre_swap is None:
            return None
        return pre_swap.item() if torch.is_tensor(pre_swap) else float(pre_swap)

    @staticmethod
    def extract_postprocess_time(batch_out):
        if len(batch_out) <= 4 or not isinstance(batch_out[4], dict):
            return None
        post_time = batch_out[4].get("postprocess_time", None)
        if post_time is None:
            post_time = batch_out[4].get("two_swap_time", None)
        if post_time is None:
            return None
        return post_time.item() if torch.is_tensor(post_time) else float(post_time)

    @staticmethod
    def lowprob_permute_matchings(matching_2d, prob_2d, m, max_cases=0):
        if prob_2d.dim() != 2 or matching_2d.dim() != 2:
            return matching_2d.unsqueeze(0)
        if prob_2d.shape[0] != matching_2d.shape[0] or prob_2d.shape[1] != matching_2d.shape[1]:
            return matching_2d.unsqueeze(0)
        matched_edges = torch.nonzero(matching_2d > 0.5, as_tuple=False)
        if matched_edges.numel() == 0:
            return matching_2d.unsqueeze(0)
        m = int(max(0, m))
        if m <= 1:
            return matching_2d.unsqueeze(0)
        m = min(m, int(matched_edges.shape[0]))
        matched_scores = prob_2d[matched_edges[:, 0], matched_edges[:, 1]]
        low_idx = torch.argsort(matched_scores, descending=False)[:m]
        sel_rows = matched_edges[low_idx, 0]
        sel_cols = matched_edges[low_idx, 1]

        candidates = []
        max_cases = int(max_cases)
        case_count = 0
        sel_cols_list = sel_cols.detach().cpu().tolist()
        for perm_cols in itertools.permutations(sel_cols_list, m):
            candidate = matching_2d.clone()
            candidate[sel_rows, :] = 0.0
            perm_cols_tensor = torch.tensor(perm_cols, device=matching_2d.device, dtype=torch.long)
            candidate[sel_rows, perm_cols_tensor] = 1.0
            candidates.append(candidate)
            case_count += 1
            if max_cases > 0 and case_count >= max_cases:
                break
        if not candidates:
            return matching_2d.unsqueeze(0)
        return torch.stack(candidates, dim=0)

    @staticmethod
    def matching_tensor_to_edge_list(matching_tensor):
        if matching_tensor.dim() == 3:
            matching_tensor = matching_tensor[0]
        edge_idx = torch.nonzero(matching_tensor > 0.5, as_tuple=False)
        return [[int(row), int(col)] for row, col in edge_idx.tolist()]

    @staticmethod
    def serialize_matchings(matching_tensor):
        if matching_tensor.dim() == 2:
            matching_tensor = matching_tensor.unsqueeze(0)
        serialized = []
        for matching in matching_tensor:
            edge_idx = torch.nonzero(matching > 0.5, as_tuple=False)
            serialized.append([[int(row), int(col)] for row, col in edge_idx.tolist()])
        return serialized

    def _expand_disjoint_batch_tensorized(self, batch, num_parallel_sampling):
        """Repeat each graph pair k times without rebuilding Data objects."""
        if num_parallel_sampling <= 1:
            return batch

        pair_count = int(batch.n.size(0))
        if pair_count == 0:
            return batch

        device = batch.x.device
        expanded_pair_count = pair_count * int(num_parallel_sampling)

        node_counts = (batch.ptr[1:] - batch.ptr[:-1]).long()
        edge_batch = batch.batch[batch.edge_index[0]].long()
        graph_edge_counts = torch.bincount(edge_batch, minlength=pair_count)
        mapping_edge_batch = batch.batch[batch.edge_index_mapping[0]].long()
        mapping_edge_counts = torch.bincount(mapping_edge_batch, minlength=pair_count)

        graph_edge_ptr = torch.zeros(pair_count + 1, dtype=torch.long, device=device)
        graph_edge_ptr[1:] = torch.cumsum(graph_edge_counts, dim=0)
        mapping_edge_ptr = torch.zeros(pair_count + 1, dtype=torch.long, device=device)
        mapping_edge_ptr[1:] = torch.cumsum(mapping_edge_counts, dim=0)

        expanded_node_counts = node_counts.repeat_interleave(num_parallel_sampling)
        expanded_ptr = torch.zeros(expanded_pair_count + 1, dtype=torch.long, device=device)
        expanded_ptr[1:] = torch.cumsum(expanded_node_counts, dim=0)

        x_chunks = []
        x_indicator_chunks = []
        edge_index_chunks = []
        mapping_edge_index_chunks = []
        mapping_attr_chunks = []
        base_cost_chunks = []

        expanded_n = batch.n.repeat_interleave(num_parallel_sampling, dim=0)
        expanded_i_j = batch.i_j.repeat_interleave(num_parallel_sampling, dim=0)
        expanded_ged = batch.ged.repeat_interleave(num_parallel_sampling, dim=0)

        for pair_idx in range(pair_count):
            node_start = int(batch.ptr[pair_idx].item())
            node_end = int(batch.ptr[pair_idx + 1].item())
            local_x = batch.x[node_start:node_end]
            local_indicator = batch.x_indicator[node_start:node_end]

            graph_edge_start = int(graph_edge_ptr[pair_idx].item())
            graph_edge_end = int(graph_edge_ptr[pair_idx + 1].item())
            local_edge_index = batch.edge_index[:, graph_edge_start:graph_edge_end] - node_start

            map_edge_start = int(mapping_edge_ptr[pair_idx].item())
            map_edge_end = int(mapping_edge_ptr[pair_idx + 1].item())
            local_mapping_edge_index = batch.edge_index_mapping[:, map_edge_start:map_edge_end] - node_start
            local_mapping_attr = batch.edge_attr_mapping[map_edge_start:map_edge_end]
            local_base_cost = batch.edge_base_cost[map_edge_start:map_edge_end]

            copy_pair_start = pair_idx * num_parallel_sampling
            copy_node_offsets = expanded_ptr[copy_pair_start:copy_pair_start + num_parallel_sampling].long()

            x_chunks.append(local_x.repeat((num_parallel_sampling, 1)))
            x_indicator_chunks.append(local_indicator.repeat((num_parallel_sampling, 1)))

            repeated_edge_index = local_edge_index.unsqueeze(1) + copy_node_offsets.view(1, -1, 1)
            edge_index_chunks.append(repeated_edge_index.reshape(2, -1))

            repeated_mapping_edge_index = local_mapping_edge_index.unsqueeze(1) + copy_node_offsets.view(1, -1, 1)
            mapping_edge_index_chunks.append(repeated_mapping_edge_index.reshape(2, -1))

            mapping_attr_chunks.append(local_mapping_attr.repeat((num_parallel_sampling, 1)))
            base_cost_chunks.append(local_base_cost.repeat((num_parallel_sampling, 1)))

        new_batch = Data()
        new_batch.x = torch.cat(x_chunks, dim=0)
        new_batch.edge_index = torch.cat(edge_index_chunks, dim=1)
        new_batch.x_indicator = torch.cat(x_indicator_chunks, dim=0)
        new_batch.edge_index_mapping = torch.cat(mapping_edge_index_chunks, dim=1)
        new_batch.edge_attr_mapping = torch.cat(mapping_attr_chunks, dim=0)
        new_batch.edge_base_cost = torch.cat(base_cost_chunks, dim=0)
        new_batch.n = expanded_n
        new_batch.i_j = expanded_i_j
        new_batch.ged = expanded_ged
        new_batch.ptr = expanded_ptr
        new_batch.batch = torch.arange(expanded_pair_count, device=device).repeat_interleave(expanded_node_counts)
        return new_batch
    
    def setup_model(self):
        if self.args.denoise_network == "lightgt_dense":
            self.model = DenseLightGTDiffMatch(self.args, self.number_of_labels).to(self.device)
        elif self.args.denoise_network == "lightgt_disjoint":
            self.model = DisjointLightGTDiffMatch(self.args, self.number_of_labels).to(self.device)
        else:
            raise ValueError(
                "Unsupported denoise network in slim build: {}. "
                "Use lightgt_dense or lightgt_disjoint.".format(self.args.denoise_network)
            )
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.use_gpu else None,
                output_device=self.local_rank if self.use_gpu else None,
                find_unused_parameters=True,
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)
    
    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))
        pair_manifest_path = os.path.join(self.args.abs_path, "json_data", dataset_name, "pair_manifest.json")
        self.has_precomputed_pairs = os.path.isfile(pair_manifest_path)
        self.precomputed_pair_splits = None
        if self.has_precomputed_pairs:
            self.precomputed_pair_splits = json.load(open(pair_manifest_path, "r"))
            print("Load {} precomputed graph pairs.".format(len(self.precomputed_pair_splits)))

        labels_path = os.path.join(self.args.abs_path, "json_data", dataset_name, "labels.json")
        self.number_of_labels = 0
        if os.path.isfile(labels_path):
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        
        ged_dict = dict()
        load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
        t2 = time.time()
        self.load_data_time = t2 - t1
    
    def transfer_data_to_torch(self):
        t1 = time.time()
        self.log_stage("transfer_data_to_torch: building edge_index list")

        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)
        
        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)
        self.log_stage("transfer_data_to_torch: features tensorized")

        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gid_to_index = {graph_gid: idx for idx, graph_gid in enumerate(gid)}
        # number of nodes
        self.gn = [g['n'] for g in self.graphs]
        # number of edges
        self.gm = [g['m'] for g in self.graphs]
        self.identity_mappings = [torch.eye(graph_n, dtype=torch.float) for graph_n in self.gn]
        self.zero_ta_ged = (0.0, 0.0, 0.0, 0.0)
        self.pair_metadata = {}
        self.log_stage("transfer_data_to_torch: building pair_metadata")
        for (gid_1, gid_2), (ta_ged, gt_mappings) in self.ged_dict.items():
            idx_1 = self.gid_to_index.get(gid_1)
            idx_2 = self.gid_to_index.get(gid_2)
            if idx_1 is None or idx_2 is None:
                continue
            self.pair_metadata[(idx_1, idx_2)] = {
                "ta_ged": ta_ged,
                "node_mappings": gt_mappings,
            }
        self.log_stage("transfer_data_to_torch: pair_metadata ready ({})".format(len(self.pair_metadata)))
        
        t2 = time.time()
        self.to_torch_time = t2 - t1

    @staticmethod
    def node_mapping_to_matrix(node_mapping, n1, n2):
        mapping_matrix = torch.zeros((n1, n2), dtype=torch.float)
        for x, y in enumerate(node_mapping):
            mapping_matrix[x, y] = 1.0
        return mapping_matrix

    def get_pair_metadata(self, id_1, id_2):
        if id_1 == id_2:
            return {
                "ta_ged": self.zero_ta_ged,
                "mapping": self.identity_mappings[id_1],
                "mapping_candidates": [self.identity_mappings[id_1]],
            }

        pair_data = self.pair_metadata.get((id_1, id_2))
        if pair_data is not None:
            n1 = self.gn[id_1]
            n2 = self.gn[id_2]
            dense_mappings = [self.node_mapping_to_matrix(gt_mapping, n1, n2) for gt_mapping in pair_data["node_mappings"]]
            return {
                "ta_ged": pair_data["ta_ged"],
                "mapping": dense_mappings[0],
                "mapping_candidates": dense_mappings,
            }

        reverse_pair_data = self.pair_metadata.get((id_2, id_1))
        if reverse_pair_data is None:
            return None

        n1 = self.gn[id_2]
        n2 = self.gn[id_1]
        reverse_dense_mappings = [self.node_mapping_to_matrix(gt_mapping, n1, n2) for gt_mapping in reverse_pair_data["node_mappings"]]
        return {
            "ta_ged": reverse_pair_data["ta_ged"],
            "mapping": reverse_dense_mappings[0].t().contiguous(),
            "mapping_candidates": [mapping.t().contiguous() for mapping in reverse_dense_mappings],
        }
    
    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense()
       
        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long()

        feature2 = torch.zeros(f.shape)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data
    
    def gen_delta_graphs(self):
        if self.has_precomputed_pairs:
            self.log_stage("gen_delta_graphs: skipped because pair_manifest is available")
            return
        random.seed(0)
        k = self.args.num_delta_graphs
        self.log_stage("gen_delta_graphs: scanning {} graphs".format(len(self.graphs)))
        large_graph_indices = [i for i, g in enumerate(self.graphs) if g['n'] > 10]
        self.log_stage(
            "gen_delta_graphs: {} large graphs need {} synthetic variants each (total {})".format(
                len(large_graph_indices), k, len(large_graph_indices) * k
            )
        )
        delta_pbar = tqdm(
            total=len(large_graph_indices),
            desc="Gen delta graphs",
            unit="graphs",
            leave=False,
            position=self.args.tqdm_position,
        ) if self.show_progress else None
        for progress_idx, i in enumerate(large_graph_indices, start=1):
            g = self.graphs[i]
            # gen k delta graphs
            f = self.features[i]
            self.delta_graphs[i] = [self.delta_graph(g, f, self.device) for j in range(k)]
            if delta_pbar is not None:
                delta_pbar.update(1)
            elif self.is_main_process and (progress_idx == 1 or progress_idx % 50 == 0 or progress_idx == len(large_graph_indices)):
                print(
                    "[Init] gen_delta_graphs: processed {}/{} large graphs".format(
                        progress_idx, len(large_graph_indices)
                    ),
                    flush=True,
                )
        if delta_pbar is not None:
            delta_pbar.close()
        self.log_stage("gen_delta_graphs: completed")
    
    def pack_graph_pair(self,pair):
        new_data = Data()
        (pair_type, id_1, id_2) = pair
        if pair_type == 0:
            new_data.i_j = torch.tensor([[id_1,id_2]])
            pair_data = self.get_pair_metadata(id_1, id_2)
            if pair_data is None:
                raise KeyError("Missing graph pair metadata for ({}, {}).".format(id_1, id_2))
            real_ged = pair_data["ta_ged"][0]

            n1 = self.gn[id_1]
            n2 = self.gn[id_2]

            new_data.n = torch.tensor([[n1,n2]])
            new_data.x = torch.cat([self.features[id_1],self.features[id_2]],dim=0)
            new_data.edge_index = torch.cat([self.edge_index[id_1],self.edge_index[id_2]+n1],dim=1)
            # (G,G'): If G, then x_indicator=0. If G', x_indicator=1
            new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

            # transfer mapping to edge index between G and G'
            mapping = pair_data["mapping"] + 0.1
            mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
            mapping_edge_index[1] += n1
            new_data.edge_index_mapping = mapping_edge_index
            new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)
            dense_base_cost = self.base_match_score_adjustment(new_data, n1, n2)
            new_data.edge_base_cost = dense_base_cost[mapping_edge_index[0], mapping_edge_index[1] - n1].unsqueeze(-1)

            new_data.ged = real_ged
        
        else:
            # synthetic graph
            new_data.i_j = torch.tensor([[id_1,id_2]])
            dg: dict = self.delta_graphs[id_1][id_2]
            real_ged = dg["ta_ged"][0]
            n1 = self.gn[id_1]
            n2 = dg["n"]
            new_data.n = torch.tensor([[n1,n2]])
            new_data.x = torch.cat([self.features[id_1],dg["features"]],dim=0)
            new_data.edge_index = torch.cat([self.edge_index[id_1],dg["edge_index"]+n1],dim=1)
            new_data.x_indicator = torch.cat([torch.zeros((n1,1)),torch.ones((n2,1))],dim=0)

            mapping =  dg["mapping"] + 0.1
            mapping_edge_index,mapping_edge_attr = dense_to_sparse(mapping)
            mapping_edge_index[1] += n1
            new_data.edge_index_mapping = mapping_edge_index
            new_data.edge_attr_mapping = (mapping_edge_attr-0.1).unsqueeze(-1)
            dense_base_cost = self.base_match_score_adjustment(new_data, n1, n2)
            new_data.edge_base_cost = dense_base_cost[mapping_edge_index[0], mapping_edge_index[1] - n1].unsqueeze(-1)
            new_data.ged = real_ged
        return new_data
    
    def check_pair(self, i, j):
        if self.get_pair_metadata(i, j) is not None:
            return (0, i, j)
        return None
    
    def init_graph_pairs(self):
        start = time.time()
        random.seed(1)
        self.log_stage("init_graph_pairs: begin")
        if self.has_precomputed_pairs:
            training_specs = []
            val_specs = []
            testing_specs = []
            missing_gids = []
            self.log_stage("init_graph_pairs: loading pair specs from manifest")
            for pair_item in self.precomputed_pair_splits:
                query_gid = int(pair_item["new_query_gid"])
                db_gid = int(pair_item["new_db_gid"])
                query_idx = self.gid_to_index.get(query_gid)
                db_idx = self.gid_to_index.get(db_gid)
                if query_idx is None or db_idx is None:
                    missing_gids.append((query_gid, db_gid))
                    continue
                pair_spec = (0, query_idx, db_idx)
                split_name = pair_item["benchmark_split"]
                if split_name == "train":
                    training_specs.append(pair_spec)
                elif split_name == "val":
                    val_specs.append(pair_spec)
                elif split_name == "test":
                    testing_specs.append(pair_spec)
                else:
                    raise ValueError("Unknown benchmark split: {}".format(split_name))
            if missing_gids:
                raise ValueError(
                    "Manifest references {} missing graph gid pairs. "
                    "First examples: {}. "
                    "Check json_data/{}/train|test contents vs pair_manifest.json.".format(
                        len(missing_gids), missing_gids[:5], self.args.dataset
                    )
                )

            self.training_graphs = GraphPairDataset(self, training_specs)
            self.val_graphs = GraphPairDataset(self, val_specs)
            self.testing_graphs = GraphPairDataset(self, testing_specs)
            self.testing_graphs_small = GraphPairDataset(self, testing_specs)
            self.testing_graphs_large = GraphPairDataset(self, testing_specs)

            end = time.time()
            print("Load {} training graph pairs from manifest.".format(len(self.training_graphs)))
            print("Load {} val graph pairs from manifest.".format(len(self.val_graphs)))
            print("Load {} testing graph pairs from manifest.".format(len(self.testing_graphs)))
            print("Load {} small testing graph pairs from manifest.".format(len(self.testing_graphs_small)))
            print("Load {} large testing graph pairs from manifest.".format(len(self.testing_graphs_large)))
            print("Generation time:",end-start)
            return

        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing_graphs_small = []
        self.testing_graphs_large = []

        dg = self.delta_graphs

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)
        self.log_stage(
            "init_graph_pairs: generating pairs for train={}, val={}, test={}".format(
                train_num, self.val_num, test_num - val_num
            )
        )

        progress_total = train_num + max(self.val_num, 0) + max(test_num - val_num, 0)
        pair_build_pbar = tqdm(
            total=progress_total,
            desc="Init graph pairs",
            unit="graphs",
            leave=False,
            position=self.args.tqdm_position,
        ) if self.show_progress else None

        # each training graph is paired with all other training graphs or 100 synthetic graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        pair = self.pack_graph_pair(tmp)
                        self.training_graphs.append(pair)
                    
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    pair = self.pack_graph_pair((1, i, j))
                    self.training_graphs.append(pair)
            if pair_build_pbar is not None:
                pair_build_pbar.update(1)
        
        # each val / testing graph is pair with 100 training graphs or synthetic graphs
        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)
        
        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j))
                    self.val_graphs.append(pair)
               
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j))
                    self.val_graphs.append(pair)
            if pair_build_pbar is not None:
                pair_build_pbar.update(1)
                   
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                i_list = []
                for j in li[:self.args.num_testing_graphs]:
                    pair = self.pack_graph_pair((0, i, j))
                    self.testing_graphs.append(pair)
                    self.testing_graphs_small.append(pair)
                    
            elif dg[i] is not None:
                k = len(dg[i])
                i_list = []
                for j in list(range(k)):
                    pair = self.pack_graph_pair((1, i, j))
                    self.testing_graphs.append(pair)
                    self.testing_graphs_large.append(pair)
            if pair_build_pbar is not None:
                pair_build_pbar.update(1)

        if pair_build_pbar is not None:
            pair_build_pbar.close()

        end = time.time()
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} val graph pairs.".format(len(self.val_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generate {} small testing graph pairs.".format(len(self.testing_graphs_small)))
        print("Generate {} large testing graph pairs.".format(len(self.testing_graphs_large)))
        print("Generation time:",end-start)

    def split_batch_pair_matchings(self, data, mapping_attr=None):
        if mapping_attr is None:
            mapping_attr = data.edge_attr_mapping
        mapping_attr = mapping_attr.view(-1)
        mapping_edge_batch = data.batch[data.edge_index_mapping[0]]
        pair_matchings = []

        for pair_idx in range(data.n.shape[0]):
            n1 = int(data.n[pair_idx, 0].item())
            n2 = int(data.n[pair_idx, 1].item())
            pair_matching = torch.zeros((n1, n2), device=mapping_attr.device, dtype=mapping_attr.dtype)
            pair_mask = mapping_edge_batch == pair_idx
            pair_edge_index = data.edge_index_mapping[:, pair_mask]
            node_offset = int(data.ptr[pair_idx].item())
            local_rows = pair_edge_index[0] - node_offset
            local_cols = pair_edge_index[1] - node_offset - n1
            pair_matching[local_rows, local_cols] = mapping_attr[pair_mask]
            pair_matchings.append(pair_matching)

        return pair_matchings

    def pair_matchings_to_edge_attr(self, data, pair_matchings):
        device = data.edge_index_mapping.device
        dtype = pair_matchings[0].dtype if pair_matchings else torch.float
        mapping_edge_batch = data.batch[data.edge_index_mapping[0]]
        edge_attr = torch.zeros((data.edge_index_mapping.shape[1], 1), device=device, dtype=dtype)

        for pair_idx, pair_matching in enumerate(pair_matchings):
            pair_mask = mapping_edge_batch == pair_idx
            pair_edge_index = data.edge_index_mapping[:, pair_mask]
            node_offset = int(data.ptr[pair_idx].item())
            local_rows = pair_edge_index[0] - node_offset
            local_cols = pair_edge_index[1] - node_offset - pair_matching.shape[0]
            edge_attr[pair_mask, 0] = pair_matching[local_rows, local_cols]

        return edge_attr

    def sample_partial_pair_matchings(self, pair_matchings, t):
        partial_matchings = []
        for pair_t, gt_matching in zip(t.tolist(), pair_matchings):
            positive_idx = torch.nonzero(gt_matching > 0.5, as_tuple=False)
            keep_ratio = float(self.diffusion.keep_ratio(pair_t))
            keep_count = int(round(keep_ratio * positive_idx.shape[0]))
            keep_count = min(max(keep_count, 0), positive_idx.shape[0])

            partial_matching = torch.zeros_like(gt_matching)
            if keep_count >= positive_idx.shape[0]:
                partial_matching = gt_matching.clone()
            elif keep_count > 0:
                keep_perm = torch.randperm(positive_idx.shape[0], device=gt_matching.device)[:keep_count]
                kept_edges = positive_idx[keep_perm]
                partial_matching[kept_edges[:, 0], kept_edges[:, 1]] = 1.0
            partial_matchings.append(partial_matching)

        return partial_matchings

    def adjust_match_scores(self, probability_matrix, data, pair_idx, pair_matching):
        score = torch.logit(probability_matrix.clamp(1e-6, 1 - 1e-6))
        n1, n2 = pair_matching.shape
        left_start = int(data.ptr[pair_idx].item())
        right_start = left_start + n1

        left_x = data.x[left_start:left_start + n1]
        right_x = data.x[right_start:right_start + n2]
        left_label = left_x.argmax(dim=-1) if left_x.dim() > 1 and left_x.size(-1) > 1 else left_x.view(-1).long()
        right_label = right_x.argmax(dim=-1) if right_x.dim() > 1 and right_x.size(-1) > 1 else right_x.view(-1).long()

        left_edge_mask = (data.edge_index[0] >= left_start) & (data.edge_index[0] < left_start + n1)
        right_edge_mask = (data.edge_index[0] >= right_start) & (data.edge_index[0] < right_start + n2)
        left_degree = torch.bincount((data.edge_index[0, left_edge_mask] - left_start).long(), minlength=n1).float()
        right_degree = torch.bincount((data.edge_index[0, right_edge_mask] - right_start).long(), minlength=n2).float()
        degree_scale = max(float(max(left_degree.max().item(), right_degree.max().item())), 1.0)

        left_sim = left_x @ right_x.t() if left_x.dim() == 2 and right_x.dim() == 2 else torch.zeros_like(score)
        label_cost = (left_label[:, None] != right_label[None, :]).float()
        degree_cost = (left_degree[:, None] - right_degree[None, :]).abs() / degree_scale

        adjusted = score
        adjusted = adjusted - self.args.match_cost_scale * (
            self.args.match_label_weight * label_cost
            + self.args.match_degree_weight * degree_cost
            - self.args.match_similarity_weight * left_sim
        )
        return adjusted

    @staticmethod
    def constrained_matching_decode(score_matrix, full_size):
        decoded = torch.zeros_like(score_matrix)
        flat_scores = score_matrix.reshape(-1)
        order = torch.argsort(flat_scores, descending=True)
        used_rows = torch.zeros(score_matrix.shape[0], device=score_matrix.device, dtype=torch.bool)
        used_cols = torch.zeros(score_matrix.shape[1], device=score_matrix.device, dtype=torch.bool)
        selected = 0

        for flat_idx in order.tolist():
            if selected >= full_size:
                break
            row = flat_idx // score_matrix.shape[1]
            col = flat_idx % score_matrix.shape[1]
            if used_rows[row] or used_cols[col]:
                continue
            decoded[row, col] = 1.0
            used_rows[row] = True
            used_cols[col] = True
            selected += 1

        return decoded

    def sample_partial_from_clean_matching(self, clean_matching, score_matrix, target_size):
        target_size = min(max(int(target_size), 0), int(clean_matching.sum().item()))
        if target_size <= 0:
            return torch.zeros_like(clean_matching)

        positive_idx = torch.nonzero(clean_matching > 0.5, as_tuple=False)
        if target_size >= positive_idx.shape[0]:
            return clean_matching.clone()

        if self.args.renoise_mode == "topk":
            pair_scores = score_matrix[clean_matching > 0.5]
            keep_idx = torch.argsort(pair_scores, descending=True)[:target_size]
        else:
            pair_scores = score_matrix[clean_matching > 0.5]
            weights = torch.softmax(pair_scores / max(float(self.args.renoise_temperature), 1e-6), dim=0)
            keep_idx = torch.multinomial(weights, target_size, replacement=False)

        kept_edges = positive_idx[keep_idx]
        partial_matching = torch.zeros_like(clean_matching)
        partial_matching[kept_edges[:, 0], kept_edges[:, 1]] = 1.0
        return partial_matching

    @staticmethod
    def uniform_edge_attr_to_dense(data, edge_attr, n1, n2, num_pairs):
        mapping_edge_idx = data.edge_index_mapping
        mapping_batch = data.batch[mapping_edge_idx[0]]
        local_rows = mapping_edge_idx[0] - mapping_batch * (n1 + n2)
        local_cols = mapping_edge_idx[1] - mapping_batch * (n1 + n2) - n1

        dense = torch.zeros((num_pairs, n1, n2), device=edge_attr.device, dtype=edge_attr.dtype)
        dense[mapping_batch, local_rows, local_cols] = edge_attr.view(-1)
        return dense

    @staticmethod
    def uniform_dense_to_edge_attr(data, dense, n1, n2):
        mapping_edge_idx = data.edge_index_mapping
        mapping_batch = data.batch[mapping_edge_idx[0]]
        local_rows = mapping_edge_idx[0] - mapping_batch * (n1 + n2)
        local_cols = mapping_edge_idx[1] - mapping_batch * (n1 + n2) - n1
        return dense[mapping_batch, local_rows, local_cols].unsqueeze(-1)

    @staticmethod
    def block_edge_attr_to_dense(data, edge_attr, block_start, block_size, n1, n2):
        mapping_edge_idx = data.edge_index_mapping
        mapping_batch = data.batch[mapping_edge_idx[0]]
        block_mask = (mapping_batch >= block_start) & (mapping_batch < block_start + block_size)
        block_batch = mapping_batch[block_mask] - block_start
        node_offsets = data.ptr[mapping_batch[block_mask]]
        local_rows = mapping_edge_idx[0, block_mask] - node_offsets
        local_cols = mapping_edge_idx[1, block_mask] - node_offsets - n1

        dense = torch.zeros((block_size, n1, n2), device=edge_attr.device, dtype=edge_attr.dtype)
        dense[block_batch, local_rows, local_cols] = edge_attr[block_mask].view(-1)
        return dense

    @staticmethod
    def block_dense_to_edge_attr(data, dense_blocks):
        mapping_edge_idx = data.edge_index_mapping
        mapping_batch = data.batch[mapping_edge_idx[0]]
        edge_attr = torch.zeros((mapping_edge_idx.shape[1], 1), device=mapping_edge_idx.device)

        for block_start, dense in dense_blocks:
            block_size, n1, _ = dense.shape
            block_mask = (mapping_batch >= block_start) & (mapping_batch < block_start + block_size)
            block_batch = mapping_batch[block_mask] - block_start
            node_offsets = data.ptr[mapping_batch[block_mask]]
            local_rows = mapping_edge_idx[0, block_mask] - node_offsets
            local_cols = mapping_edge_idx[1, block_mask] - node_offsets - n1
            edge_attr[block_mask, 0] = dense[block_batch, local_rows, local_cols]

        return edge_attr

    @staticmethod
    def build_block_dense_lookup(data, block_start, block_size, n1, n2):
        mapping_edge_idx = data.edge_index_mapping
        mapping_batch = data.batch[mapping_edge_idx[0]]
        block_mask = (mapping_batch >= block_start) & (mapping_batch < block_start + block_size)
        edge_indices = torch.nonzero(block_mask, as_tuple=False).view(-1)
        block_batch = mapping_batch[block_mask] - block_start
        node_offsets = data.ptr[mapping_batch[block_mask]]
        local_rows = mapping_edge_idx[0, block_mask] - node_offsets
        local_cols = mapping_edge_idx[1, block_mask] - node_offsets - n1
        flat_indices = block_batch * (n1 * n2) + local_rows * n2 + local_cols
        return edge_indices, flat_indices.long()

    @staticmethod
    def grouped_edge_attr_to_dense(edge_attr, group_spec):
        flat_dense = edge_attr.new_zeros(group_spec["total_batches"] * group_spec["n1"] * group_spec["n2"])
        flat_dense[group_spec["flat_indices"]] = edge_attr[group_spec["edge_indices"]].view(-1)
        return flat_dense.view(group_spec["total_batches"], group_spec["n1"], group_spec["n2"])

    @staticmethod
    def grouped_dense_blocks_to_edge_attr(num_edges, group_specs, dense_blocks, reference_tensor):
        edge_attr = reference_tensor.new_zeros((num_edges, 1))
        for group_spec, dense in zip(group_specs, dense_blocks):
            edge_attr[group_spec["edge_indices"], 0] = dense.reshape(-1)[group_spec["flat_indices"]]
        return edge_attr

    @staticmethod
    def build_global_padded_decode_spec(decode_groups, eval_batch_size, num_parallel_sampling, device):
        if not decode_groups:
            raise ValueError("decode_groups is empty, cannot build global padded decode spec.")

        pad_n1 = max(int(group_spec["n1"]) for group_spec in decode_groups)
        pad_n2 = max(int(group_spec["n2"]) for group_spec in decode_groups)
        total_batches = int(eval_batch_size) * int(num_parallel_sampling)
        padded_area = pad_n1 * pad_n2

        edge_chunks = []
        flat_chunks = []
        base_score_padded = torch.zeros((eval_batch_size, pad_n1, pad_n2), device=device)
        valid_mask_padded = torch.zeros((eval_batch_size, pad_n1, pad_n2), dtype=torch.bool, device=device)
        pair_max_matching_size = torch.zeros((eval_batch_size,), dtype=torch.long, device=device)

        for group_spec in decode_groups:
            n1 = int(group_spec["n1"])
            n2 = int(group_spec["n2"])
            pair_indices_tensor = torch.tensor(group_spec["pair_indices"], device=device, dtype=torch.long)
            area = n1 * n2

            local_flat = group_spec["flat_indices"]
            sample_local = torch.div(local_flat, area, rounding_mode="floor")
            rem = local_flat - sample_local * area
            local_rows = torch.div(rem, n2, rounding_mode="floor")
            local_cols = rem - local_rows * n2
            group_offset = torch.div(sample_local, num_parallel_sampling, rounding_mode="floor")
            sample_idx = sample_local - group_offset * num_parallel_sampling
            pair_ids = pair_indices_tensor[group_offset]
            global_sample = pair_ids * num_parallel_sampling + sample_idx
            padded_flat = global_sample * padded_area + local_rows * pad_n2 + local_cols

            edge_chunks.append(group_spec["edge_indices"])
            flat_chunks.append(padded_flat.long())

            for group_offset_idx, pair_idx in enumerate(group_spec["pair_indices"]):
                base_score_padded[pair_idx, :n1, :n2] = group_spec["base_score_costs"][group_offset_idx]
                valid_mask_padded[pair_idx, :n1, :n2] = True
                pair_max_matching_size[pair_idx] = int(group_spec["max_matching_size"])

        return {
            "group_spec": {
                "pair_indices": list(range(eval_batch_size)),
                "group_size": eval_batch_size,
                "total_batches": total_batches,
                "n1": pad_n1,
                "n2": pad_n2,
                "max_matching_size": int(pair_max_matching_size.max().item()) if pair_max_matching_size.numel() > 0 else 0,
                "edge_indices": torch.cat(edge_chunks, dim=0),
                "flat_indices": torch.cat(flat_chunks, dim=0),
                "base_score_costs": base_score_padded,
            },
            "pair_valid_mask": valid_mask_padded,
            "pair_max_matching_size": pair_max_matching_size,
            "pad_n1": pad_n1,
            "pad_n2": pad_n2,
        }

    @staticmethod
    def ged_values_from_clean_matchings(data, clean_matchings, n1, n2):
        num_parallel_sampling = clean_matchings.shape[0]
        solution = clean_matchings.bool()
        max_nodes = max(n1, n2)
        square_solution = torch.zeros(
            (num_parallel_sampling, max_nodes, max_nodes),
            device=solution.device,
            dtype=torch.bool,
        )
        square_solution[:, :n1, :n2] = solution

        for sample_idx in range(num_parallel_sampling):
            sample = square_solution[sample_idx]
            row_has_match = sample.any(dim=1)
            col_has_match = sample.any(dim=0)
            unmatched_rows = torch.where(~row_has_match)[0]
            unmatched_cols = torch.where(~col_has_match)[0]
            fill_cnt = min(unmatched_rows.shape[0], unmatched_cols.shape[0])
            if fill_cnt > 0:
                sample[unmatched_rows[:fill_cnt], unmatched_cols[:fill_cnt]] = True

        perm = torch.argmax(square_solution.long(), dim=-1)
        x1 = data.x[:n1]
        x2 = data.x[n1:n1 + n2]
        dense_x1 = torch.zeros((num_parallel_sampling, max_nodes, x1.shape[-1]), device=x1.device, dtype=x1.dtype)
        dense_x2 = torch.zeros((num_parallel_sampling, max_nodes, x2.shape[-1]), device=x2.device, dtype=x2.dtype)
        dense_x1[:, :n1, :] = x1.unsqueeze(0).expand(num_parallel_sampling, -1, -1)
        dense_x2[:, :n2, :] = x2.unsqueeze(0).expand(num_parallel_sampling, -1, -1)
        permuted_x2 = dense_x2.gather(1, perm.unsqueeze(-1).expand(-1, -1, dense_x2.shape[-1]))

        edge1 = data.edge_index[:, data.edge_index[0] < n1]
        edge1 = remove_self_loops(edge1)[0]
        dense_adj_1 = to_dense_adj(edge_index=edge1, max_num_nodes=max_nodes)[0]

        edge2 = data.edge_index[:, data.edge_index[0] >= n1] - n1
        dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0], max_num_nodes=max_nodes)[0]
        dense_adj_2 = dense_adj_2.unsqueeze(0).expand(num_parallel_sampling, -1, -1)
        dense_adj_2 = dense_adj_2.gather(1, perm.unsqueeze(-1).expand(-1, -1, max_nodes))
        dense_adj_2 = dense_adj_2.gather(2, perm.unsqueeze(1).expand(-1, max_nodes, -1))

        adj_diff = torch.abs(dense_adj_1.unsqueeze(0) - dense_adj_2).view(num_parallel_sampling, -1).sum(dim=-1) // 2
        feat_diff = torch.sum(~torch.all(dense_x1 == permuted_x2, dim=-1), dim=-1)
        ged = adj_diff + feat_diff
        return ged, square_solution[:, :n1, :n2]

    @staticmethod
    def ged_from_clean_matchings(data, clean_matchings, n1, n2):
        ged, canonical_matchings = Trainer.ged_values_from_clean_matchings(data, clean_matchings, n1, n2)

        min_ged = ged.min()
        min_mapping = canonical_matchings[min_ged == ged]
        return min_ged, min_mapping

    @staticmethod
    def _matching_to_assignment(matching_2d):
        row_has_match = matching_2d.any(dim=1)
        if not bool(row_has_match.all()):
            return None
        assignment = torch.argmax(matching_2d.long(), dim=1)
        if assignment.unique().numel() != assignment.numel():
            return None
        return assignment.long()

    @staticmethod
    def _assignments_to_clean_matchings(assignments, n1, n2):
        num_samples = assignments.shape[0]
        clean = torch.zeros((num_samples, n1, n2), device=assignments.device, dtype=torch.float32)
        row_idx = torch.arange(n1, device=assignments.device).unsqueeze(0).expand(num_samples, -1)
        clean[torch.arange(num_samples, device=assignments.device).unsqueeze(1), row_idx, assignments] = 1.0
        return clean

    @staticmethod
    def ged_values_from_assignments(data, assignments, n1, n2):
        clean = Trainer._assignments_to_clean_matchings(assignments, n1, n2)
        ged, _ = Trainer.ged_values_from_clean_matchings(data, clean, n1, n2)
        return ged

    def batched_two_swap_ged_search(
        self,
        data,
        init_matching,
        max_iter=50,
        eps=1e-9,
        chunk_size=None,
    ):
        if init_matching.dim() != 2:
            raise ValueError("init_matching must have shape [B, n].")
        B, n = init_matching.shape
        if n < 2:
            cur_ged = self.ged_values_from_assignments(data, init_matching, n, n)
            return init_matching, cur_ged, 0, []

        pairs = torch.triu_indices(n, n, offset=1, device=init_matching.device)
        i_idx, j_idx = pairs[0], pairs[1]
        S = i_idx.numel()
        if chunk_size is None or chunk_size <= 0:
            chunk_size = S

        matching = init_matching.clone()
        cur_ged = self.ged_values_from_assignments(data, matching, n, n)
        history = []

        for it in range(int(max_iter)):
            best_ged = torch.full_like(cur_ged, float("inf"))
            best_swap_idx = torch.zeros((B,), device=matching.device, dtype=torch.long)
            batch_idx = torch.arange(B, device=matching.device)[:, None]

            for start in range(0, S, chunk_size):
                end = min(start + chunk_size, S)
                c = end - start
                i_chunk = i_idx[start:end]
                j_chunk = j_idx[start:end]

                candidates = matching[:, None, :].expand(B, c, n).clone()
                swap_idx = torch.arange(c, device=matching.device)[None, :]
                tmp = candidates[batch_idx, swap_idx, i_chunk].clone()
                candidates[batch_idx, swap_idx, i_chunk] = candidates[batch_idx, swap_idx, j_chunk]
                candidates[batch_idx, swap_idx, j_chunk] = tmp

                flat_candidates = candidates.reshape(B * c, n)
                ged_flat = self.ged_values_from_assignments(data, flat_candidates, n, n)
                ged_chunk = ged_flat.view(B, c)
                chunk_best_ged, chunk_best_idx = ged_chunk.min(dim=1)
                improved_chunk = chunk_best_ged < best_ged
                best_ged = torch.where(improved_chunk, chunk_best_ged, best_ged)
                best_swap_idx = torch.where(improved_chunk, chunk_best_idx + start, best_swap_idx)

            accept = best_ged < (cur_ged - float(eps))
            if not bool(accept.any()):
                break

            best_i = i_idx[best_swap_idx]
            best_j = j_idx[best_swap_idx]
            new_matching = matching.clone()
            b_idx = torch.arange(B, device=matching.device)
            tmp = new_matching[b_idx, best_i].clone()
            new_matching[b_idx, best_i] = new_matching[b_idx, best_j]
            new_matching[b_idx, best_j] = tmp

            matching = torch.where(accept[:, None], new_matching, matching)
            cur_ged = torch.where(accept, best_ged, cur_ged)
            history.append(
                {
                    "iter": it + 1,
                    "improved_count": int(accept.sum().item()),
                    "avg_ged": float(cur_ged.float().mean().item()),
                }
            )

        return matching, cur_ged, len(history), history

    def base_match_score_adjustment(self, data, n1, n2):
        left_x = data.x[:n1]
        right_x = data.x[n1:n1 + n2]
        left_label = left_x.argmax(dim=-1) if left_x.dim() > 1 and left_x.size(-1) > 1 else left_x.view(-1).long()
        right_label = right_x.argmax(dim=-1) if right_x.dim() > 1 and right_x.size(-1) > 1 else right_x.view(-1).long()

        left_edge_mask = data.edge_index[0] < n1
        right_edge_mask = data.edge_index[0] >= n1
        left_degree = torch.bincount(data.edge_index[0, left_edge_mask].long(), minlength=n1).float()
        right_degree = torch.bincount((data.edge_index[0, right_edge_mask] - n1).long(), minlength=n2).float()
        degree_scale = torch.maximum(left_degree.max(), right_degree.max()).clamp(min=1.0)

        left_sim = left_x @ right_x.t() if left_x.dim() == 2 and right_x.dim() == 2 else torch.zeros((n1, n2), device=data.x.device)
        label_cost = (left_label[:, None] != right_label[None, :]).float()
        degree_cost = (left_degree[:, None] - right_degree[None, :]).abs() / degree_scale
        return self.args.match_cost_scale * (
            self.args.match_label_weight * label_cost
            + self.args.match_degree_weight * degree_cost
            - self.args.match_similarity_weight * left_sim
        )

    @staticmethod
    def batched_constrained_matching_decode(score_matrices, full_size):
        num_pairs, n1, n2 = score_matrices.shape
        work_scores = score_matrices.clone()
        decoded = torch.zeros_like(work_scores)
        batch_idx = torch.arange(num_pairs, device=work_scores.device)

        for _ in range(full_size):
            argmax_result = torch.argmax(work_scores.view(num_pairs, -1), dim=-1)
            rows = argmax_result // n2
            cols = argmax_result % n2

            decoded[batch_idx, rows, cols] = 1.0
            work_scores[batch_idx, rows, :] = float("-inf")
            work_scores[batch_idx, :, cols] = float("-inf")

        return decoded

    @staticmethod
    def batched_constrained_matching_decode_variable_size(score_matrices, full_sizes):
        num_pairs, _, n2 = score_matrices.shape
        work_scores = score_matrices.clone()
        decoded = torch.zeros_like(work_scores)
        batch_idx = torch.arange(num_pairs, device=work_scores.device)
        full_sizes = full_sizes.long()
        if full_sizes.numel() == 0:
            return decoded

        max_steps = int(full_sizes.max().item()) if full_sizes.numel() > 0 else 0
        selected_counts = torch.zeros_like(full_sizes)

        for _ in range(max_steps):
            active = selected_counts < full_sizes
            if not bool(active.any()):
                break

            flat_scores = work_scores.view(num_pairs, -1)
            argmax_result = torch.argmax(flat_scores, dim=-1)
            best_scores = flat_scores[batch_idx, argmax_result]
            active = active & torch.isfinite(best_scores)
            if not bool(active.any()):
                break

            rows = argmax_result // n2
            cols = argmax_result % n2
            valid_batch = batch_idx[active]
            valid_rows = rows[active]
            valid_cols = cols[active]

            decoded[valid_batch, valid_rows, valid_cols] = 1.0
            work_scores[valid_batch, valid_rows, :] = float("-inf")
            work_scores[valid_batch, :, valid_cols] = float("-inf")
            selected_counts[active] += 1

        return decoded

    def batched_constrained_matching_decode_by_mode(self, score_matrices, full_size):
        mode = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
        if mode == "conf_row_greedy_n2":
            return self.batched_confidence_ordered_row_greedy(score_matrices)
        if mode == "row_top1_unique_n2":
            return self.batched_row_top1_unique(score_matrices)
        return self.batched_constrained_matching_decode(score_matrices, full_size)

    def batched_row_top1_unique(self, score_matrices):
        # row_top1_unique_n2:
        # 1) each row proposes top-1 column
        # 2) each column keeps only the highest-scoring proposer row
        num_pairs, n1, n2 = score_matrices.shape
        decoded = torch.zeros_like(score_matrices)
        if n1 == 0 or n2 == 0:
            return decoded

        # [B, N1], [B, N1]
        row_best_scores, row_best_cols = torch.max(score_matrices, dim=2)
        valid = torch.isfinite(row_best_scores)
        if not bool(valid.any()):
            return decoded

        # For each (batch, col), keep max row score among proposers.
        col_best_scores = torch.full(
            (num_pairs, n2),
            float("-inf"),
            device=score_matrices.device,
            dtype=score_matrices.dtype,
        )
        safe_scores = row_best_scores.masked_fill(~valid, float("-inf"))
        col_best_scores.scatter_reduce_(
            1,
            row_best_cols,
            safe_scores,
            reduce="amax",
            include_self=True,
        )

        winners = valid & (row_best_scores == col_best_scores.gather(1, row_best_cols))
        if not bool(winners.any()):
            return decoded

        # Tie-break by smaller row index within each (batch, col).
        row_ids = torch.arange(n1, device=score_matrices.device, dtype=torch.long).unsqueeze(0).expand(num_pairs, -1)
        sentinel = torch.full_like(row_ids, n1)
        candidate_rows = torch.where(winners, row_ids, sentinel)
        col_best_rows = torch.full((num_pairs, n2), n1, device=score_matrices.device, dtype=torch.long)
        col_best_rows.scatter_reduce_(
            1,
            row_best_cols,
            candidate_rows,
            reduce="amin",
            include_self=True,
        )

        final_winners = winners & (row_ids == col_best_rows.gather(1, row_best_cols))
        batch_idx, row_idx = torch.nonzero(final_winners, as_tuple=True)
        col_idx = row_best_cols[batch_idx, row_idx]
        decoded[batch_idx, row_idx, col_idx] = 1.0
        return decoded

    @staticmethod
    def _row_confidence_order(row_scores):
        if row_scores.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=row_scores.device)
        topk = torch.topk(row_scores, k=min(2, row_scores.shape[1]), dim=1).values
        if row_scores.shape[1] >= 2:
            confidence = topk[:, 0] - topk[:, 1]
        else:
            confidence = topk[:, 0]
        return torch.argsort(confidence, descending=True)

    @classmethod
    def _batched_row_confidence(cls, score_matrices, invalid_row_mask=None):
        if score_matrices.shape[1] == 0:
            return score_matrices.new_empty((score_matrices.shape[0], 0))
        topk = torch.topk(score_matrices, k=min(2, score_matrices.shape[2]), dim=-1).values
        if score_matrices.shape[2] >= 2:
            confidence = topk[..., 0] - topk[..., 1]
        else:
            confidence = topk[..., 0]
        if invalid_row_mask is not None:
            confidence = confidence.masked_fill(invalid_row_mask, float("-inf"))
        return confidence

    def _batched_fill_partial_with_row_greedy(self, score_matrices, decoded, used_rows, used_cols, target_size):
        num_pairs, n1, _ = score_matrices.shape
        if target_size <= 0 or n1 == 0:
            return decoded

        fill_start = self.start_timer(sync=self.args.timing_breakdown)
        batch_idx = torch.arange(num_pairs, device=score_matrices.device)
        selected_counts = decoded.sum(dim=(1, 2)).long()

        for _ in range(target_size):
            need_fill = selected_counts < target_size
            if not bool(need_fill.any()):
                break

            masked_scores = score_matrices.masked_fill(used_cols.unsqueeze(1), float("-inf"))
            row_conf = self._batched_row_confidence(masked_scores, invalid_row_mask=used_rows)
            best_rows = torch.argmax(row_conf, dim=1)
            best_row_conf = row_conf[batch_idx, best_rows]
            valid = need_fill & torch.isfinite(best_row_conf)
            if not bool(valid.any()):
                break

            row_scores = masked_scores[batch_idx, best_rows, :]
            best_cols = torch.argmax(row_scores, dim=1)
            best_col_scores = row_scores[batch_idx, best_cols]
            valid = valid & torch.isfinite(best_col_scores)
            if not bool(valid.any()):
                break

            valid_batch = batch_idx[valid]
            valid_rows = best_rows[valid]
            valid_cols = best_cols[valid]
            decoded[valid_batch, valid_rows, valid_cols] = 1.0
            used_rows[valid_batch, valid_rows] = True
            used_cols[valid_batch, valid_cols] = True
            selected_counts[valid] += 1

        self.stop_timer(fill_start, "decode/decoder_a_mutual_fill", sync=self.args.timing_breakdown)
        return decoded

    def batched_mutual_topk_partial_decode(self, score_matrices, target_size, k):
        num_pairs, n1, n2 = score_matrices.shape
        target_size = min(max(int(target_size), 0), min(n1, n2))
        decoded = torch.zeros_like(score_matrices)
        if target_size <= 0:
            return decoded

        build_start = self.start_timer(sync=self.args.timing_breakdown)
        k_row = min(max(int(k), 1), n2)
        k_col = min(max(int(k), 1), n1)
        row_topk = torch.topk(score_matrices, k=k_row, dim=-1)
        row_topk_scores = row_topk.values
        row_topk_cols = row_topk.indices
        col_topk_rows = torch.topk(score_matrices.transpose(1, 2), k=k_col, dim=-1).indices

        batch_ids = torch.arange(num_pairs, device=score_matrices.device)[:, None, None]
        row_ids = torch.arange(n1, device=score_matrices.device)[None, :, None]
        candidate_col_top_rows = col_topk_rows[batch_ids, row_topk_cols]
        mutual_mask = (candidate_col_top_rows == row_ids.unsqueeze(-1)).any(dim=-1)
        self.stop_timer(build_start, "decode/decoder_a_mutual_build", sync=self.args.timing_breakdown)

        resolve_start = self.start_timer(sync=self.args.timing_breakdown)
        candidate_scores = row_topk_scores.masked_fill(~mutual_mask, float("-inf"))
        candidate_cols = row_topk_cols
        used_rows = torch.zeros((num_pairs, n1), dtype=torch.bool, device=score_matrices.device)
        used_cols = torch.zeros((num_pairs, n2), dtype=torch.bool, device=score_matrices.device)
        selected_counts = torch.zeros(num_pairs, dtype=torch.long, device=score_matrices.device)
        row_ids_2d = torch.arange(n1, device=score_matrices.device)[None, :].expand(num_pairs, -1)

        for _ in range(target_size):
            active = selected_counts < target_size
            if not bool(active.any()):
                break

            col_used_mask = used_cols.gather(1, candidate_cols.reshape(num_pairs, -1)).reshape(num_pairs, n1, k_row)
            masked_candidate_scores = candidate_scores.masked_fill(
                used_rows.unsqueeze(-1) | col_used_mask,
                float("-inf"),
            )

            row_best_scores, row_best_pos = torch.max(masked_candidate_scores, dim=-1)
            row_best_cols = candidate_cols.gather(2, row_best_pos.unsqueeze(-1)).squeeze(-1)
            valid_rows = active[:, None] & torch.isfinite(row_best_scores) & (~used_rows)
            if not bool(valid_rows.any()):
                break

            safe_row_scores = row_best_scores.masked_fill(~valid_rows, float("-inf"))
            col_best_scores = torch.full(
                (num_pairs, n2),
                float("-inf"),
                device=score_matrices.device,
                dtype=score_matrices.dtype,
            )
            col_best_scores.scatter_reduce_(
                1,
                row_best_cols,
                safe_row_scores,
                reduce="amax",
                include_self=True,
            )

            score_winners = valid_rows & (safe_row_scores == col_best_scores.gather(1, row_best_cols))
            if not bool(score_winners.any()):
                break

            candidate_row_ids = row_ids_2d.masked_fill(~score_winners, n1)
            col_best_rows = torch.full(
                (num_pairs, n2),
                n1,
                device=score_matrices.device,
                dtype=torch.long,
            )
            col_best_rows.scatter_reduce_(
                1,
                row_best_cols,
                candidate_row_ids,
                reduce="amin",
                include_self=True,
            )

            accepted = score_winners & (row_ids_2d == col_best_rows.gather(1, row_best_cols))
            if not bool(accepted.any()):
                break

            accepted_batch, accepted_rows = torch.nonzero(accepted, as_tuple=True)
            accepted_cols = row_best_cols[accepted_batch, accepted_rows]
            decoded[accepted_batch, accepted_rows, accepted_cols] = 1.0
            used_rows[accepted_batch, accepted_rows] = True
            used_cols[accepted_batch, accepted_cols] = True
            selected_counts = selected_counts + accepted.sum(dim=1)

        self.stop_timer(resolve_start, "decode/decoder_a_mutual_resolve", sync=self.args.timing_breakdown)
        if bool((selected_counts < target_size).any()):
            decoded = self._batched_fill_partial_with_row_greedy(
                score_matrices,
                decoded,
                used_rows,
                used_cols,
                target_size,
            )

        return decoded

    def batched_confidence_ordered_row_greedy(self, score_matrices, profile_key=None):
        greedy_start = None
        if profile_key is not None:
            greedy_start = self.start_timer(sync=self.args.timing_breakdown)
        num_pairs, n1, n2 = score_matrices.shape
        transposed = False
        work_scores = score_matrices
        if n1 > n2:
            work_scores = score_matrices.transpose(1, 2)
            num_pairs, n1, n2 = work_scores.shape
            transposed = True

        decoded = torch.zeros_like(work_scores)
        used_cols = torch.zeros((num_pairs, n2), dtype=torch.bool, device=work_scores.device)
        row_conf = self._batched_row_confidence(work_scores)
        row_order = torch.argsort(row_conf, dim=1, descending=True)
        batch_idx = torch.arange(num_pairs, device=work_scores.device)

        for step in range(n1):
            rows = row_order[:, step]
            row_scores = work_scores[batch_idx, rows, :].masked_fill(used_cols, float("-inf"))
            best_scores, cols = torch.max(row_scores, dim=1)
            valid = torch.isfinite(best_scores)
            if not bool(valid.any()):
                break

            valid_batch = batch_idx[valid]
            valid_rows = rows[valid]
            valid_cols = cols[valid]
            decoded[valid_batch, valid_rows, valid_cols] = 1.0
            used_cols[valid_batch, valid_cols] = True

        decoded = decoded.transpose(1, 2) if transposed else decoded
        if greedy_start is not None:
            self.stop_timer(greedy_start, profile_key, sync=self.args.timing_breakdown)
        return decoded

    @staticmethod
    def _matched_row_confidences(clean_matching, score_matrix):
        matched_cols = torch.argmax(clean_matching, dim=1)
        has_match = clean_matching.sum(dim=1) > 0.5
        row_scores = score_matrix.clone()
        row_scores[~has_match] = float("-inf")
        selected_scores = score_matrix[torch.arange(score_matrix.shape[0], device=score_matrix.device), matched_cols]
        alternative_scores = row_scores.scatter(
            1,
            matched_cols.unsqueeze(1),
            torch.full((score_matrix.shape[0], 1), float("-inf"), device=score_matrix.device, dtype=score_matrix.dtype),
        ).max(dim=1).values
        confidence = selected_scores - alternative_scores
        confidence[~has_match] = float("inf")
        return confidence

    @classmethod
    def _residual_repair_single(cls, clean_matching, score_matrix, repair_size):
        if repair_size <= 1:
            return clean_matching

        matched_rows = torch.nonzero(clean_matching.sum(dim=1) > 0.5, as_tuple=False).view(-1)
        if matched_rows.numel() <= 1:
            return clean_matching

        row_confidence = cls._matched_row_confidences(clean_matching, score_matrix)
        num_rows = min(int(repair_size), matched_rows.numel())
        repair_rows = torch.argsort(row_confidence, descending=False)[:num_rows]
        repair_rows = repair_rows[torch.isfinite(row_confidence[repair_rows])]
        if repair_rows.numel() <= 1:
            return clean_matching

        repaired = clean_matching.clone()
        matched_cols = torch.argmax(repaired, dim=1)
        improved = True
        while improved:
            improved = False
            for idx_a in range(repair_rows.numel()):
                row_a = repair_rows[idx_a].item()
                col_a = matched_cols[row_a].item()
                for idx_b in range(idx_a + 1, repair_rows.numel()):
                    row_b = repair_rows[idx_b].item()
                    col_b = matched_cols[row_b].item()
                    current_score = score_matrix[row_a, col_a] + score_matrix[row_b, col_b]
                    swapped_score = score_matrix[row_a, col_b] + score_matrix[row_b, col_a]
                    if swapped_score > current_score:
                        repaired[row_a, col_a] = 0.0
                        repaired[row_b, col_b] = 0.0
                        repaired[row_a, col_b] = 1.0
                        repaired[row_b, col_a] = 1.0
                        matched_cols[row_a] = col_b
                        matched_cols[row_b] = col_a
                        improved = True
        return repaired

    def batched_residual_repair(self, clean_matchings, score_matrices, repair_size):
        repair_start = self.start_timer(sync=self.args.timing_breakdown)
        repaired = clean_matchings.clone()
        for pair_idx in range(clean_matchings.shape[0]):
            repaired[pair_idx] = self._residual_repair_single(
                repaired[pair_idx],
                score_matrices[pair_idx],
                repair_size,
            )
        self.stop_timer(repair_start, "decode/decoder_a_repair", sync=self.args.timing_breakdown)
        return repaired

    def batched_decoder_a_step(self, score_matrices, target_size, t2):
        if t2 == 0:
            clean_matchings = self.batched_confidence_ordered_row_greedy(
                score_matrices,
                profile_key="decode/decoder_a_final_greedy",
            )
            if self.args.decoder_a_repair:
                clean_matchings = self.batched_residual_repair(
                    clean_matchings,
                    score_matrices,
                    self.args.decoder_a_repair_size,
                )
            next_partial_matchings = clean_matchings.clone()
        else:
            next_partial_matchings = self.batched_mutual_topk_partial_decode(
                score_matrices,
                target_size,
                self.args.decoder_a_mutual_topk,
            )
            clean_matchings = next_partial_matchings
        return clean_matchings, next_partial_matchings

    def batched_sample_partial_from_clean_matching(self, clean_matchings, score_matrices, target_size):
        if target_size <= 0:
            return torch.zeros_like(clean_matchings)
        if target_size >= min(clean_matchings.shape[1], clean_matchings.shape[2]):
            return clean_matchings.clone()

        num_pairs, n1, n2 = clean_matchings.shape
        flat_clean = clean_matchings.reshape(num_pairs, -1) > 0.5
        flat_scores = score_matrices.reshape(num_pairs, -1)

        if self.args.renoise_mode == "topk":
            masked_scores = flat_scores.masked_fill(~flat_clean, float("-inf"))
            keep_idx = torch.topk(masked_scores, target_size, dim=1).indices
        else:
            masked_scores = flat_scores.masked_fill(~flat_clean, float("-inf"))
            weights = torch.softmax(masked_scores / max(float(self.args.renoise_temperature), 1e-6), dim=1)
            keep_idx = torch.multinomial(weights, target_size, replacement=False)

        partial = torch.zeros_like(flat_scores)
        partial.scatter_(1, keep_idx, 1.0)
        return partial.view(num_pairs, n1, n2)
    
    def fit(self):
        if self.training_sampler is not None:
            self.training_sampler.set_epoch(self.cur_epoch)
        if self.is_main_process:
            print("\nModel training.\n")
            if not self.show_progress:
                print(f"Start epoch {self.cur_epoch + 1}/{self.args.model_epoch_end}")
        self.reset_timing_stats()
        self.reset_candidate_pruning_stats()
        self.reset_step_ged_stats()
        if self.module_timing_profiler is not None:
            self.module_timing_profiler.reset()
        epoch_timer = self.start_timer(sync=self.args.timing_breakdown)

        self.model.train()
        iterator = self.training_data_loader
        pbar = tqdm(
            total=min(
                len(self.training_graphs),
                self.args.batch_size * self.args.max_train_batches,
            ) if self.args.max_train_batches > 0 else len(self.training_graphs),
            unit="graph_pairs",
            leave=True,
            desc="Epoch",
            file=sys.stdout,
            position=self.args.tqdm_position,
            dynamic_ncols=True,
        ) if self.show_progress else None
        loss_sum = 0.0
        main_index = 0
        index = 0
        for batch in iterator:
            transfer_start = self.start_timer(sync=self.args.timing_breakdown)
            batch = batch.to(self.device)
            self.stop_timer(transfer_start, "train/batch_to_device", sync=self.args.timing_breakdown)
            batch_total_loss = self.process_batch(batch)
            batch_pairs = (torch.max(batch.batch) + 1).item()
            loss_sum += batch_total_loss * batch_pairs
            main_index += batch_pairs
            if pbar is not None:
                loss = loss_sum / max(main_index, 1)
                pbar.update(len(batch))
                pbar.set_description(
                    "Epoch_{}: loss={} - Batch_{}: loss={}".format(
                        self.cur_epoch + 1,
                        round(1000 * loss, 3),
                        index,
                        round(1000 * batch_total_loss, 3),
                    )
                )
            index += 1
            if self.args.max_train_batches > 0 and index >= self.args.max_train_batches:
                if self.is_main_process:
                    print(
                        f"[Profiling] Early stop after {index} training batches due to --max-train-batches={self.args.max_train_batches}",
                        flush=True,
                    )
                break
        if pbar is not None:
            pbar.close()
        self.stop_timer(epoch_timer, "train/total", sync=self.args.timing_breakdown)

        train_stats = {"loss_sum": loss_sum, "count": main_index}
        gathered_stats = self._gather_objects(train_stats)
        total_loss_sum = sum(item["loss_sum"] for item in gathered_stats)
        total_count = sum(item["count"] for item in gathered_stats)
        training_loss = round(1000 * (total_loss_sum / max(total_count, 1)), 3)
        aggregated_timing = self.aggregate_timing_breakdown()
        aggregated_module_timing = self.aggregate_module_timing_breakdown() if self.module_timing_profiler is not None else None
        aggregated_candidate_pruning = self.aggregate_candidate_pruning_stats()
        training_time = aggregated_timing.get("train/total", {}).get(
            "max_s",
            self.timing_totals.get("train/total", 0.0),
        )
        
        self.results.append(
            ('model_name', 'dataset', 'graph_set', "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss))

        if self.is_main_process:
            print(*self.results[-2], sep='\t')
            print(*self.results[-1], sep='\t')
            self.print_training_timing_breakdown(total_count, aggregated=aggregated_timing)
            self.print_candidate_pruning_stats(aggregated=aggregated_candidate_pruning)
            if aggregated_module_timing:
                self.print_module_timing_breakdown(total_count, aggregated=aggregated_module_timing)
                self.save_module_timing_artifacts(total_count, aggregated=aggregated_module_timing)
        
    
    def process_batch(self,batch):
        batch_size = (torch.max(batch.batch) + 1).item()
        zero_grad_start = self.start_timer(sync=self.args.timing_breakdown)
        self.optimizer.zero_grad()
        self.stop_timer(zero_grad_start, "train/zero_grad", sync=self.args.timing_breakdown)

        gt_mapping_label = batch.edge_attr_mapping
        sample_t_start = self.start_timer(sync=False)
        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        self.stop_timer(sample_t_start, "train/sample_timesteps")

        if self.args.denoise_network == "lightgt_disjoint":
            disjoint_model = self.unwrap_model()
            diffuse_start = self.start_timer(sync=False)
            diffused_mapping = disjoint_model.sample_partial_edge_attr(
                batch,
                gt_mapping_label,
                t,
                self.diffusion.keep_ratio,
            )
            self.stop_timer(diffuse_start, "train/sample_partial_sparse")

            t = torch.from_numpy(t).float()
            forward_start = self.start_timer(sync=self.args.timing_breakdown)
            pred_mapping_label = self.model(batch, diffused_mapping.to(self.device), t.to(self.device))
            self.stop_timer(forward_start, "train/model_forward", sync=self.args.timing_breakdown)
            self.update_candidate_pruning_stats(self.unwrap_model().pop_last_topk_pruning_stats())

            loss_start = self.start_timer(sync=self.args.timing_breakdown)
            losses = mapping_loss(pred_mapping_label, batch)
            self.stop_timer(loss_start, "train/loss", sync=self.args.timing_breakdown)

            backward_start = self.start_timer(sync=self.args.timing_breakdown)
            losses.backward()
            self.stop_timer(backward_start, "train/backward", sync=self.args.timing_breakdown)

            step_start = self.start_timer(sync=self.args.timing_breakdown)
            self.optimizer.step()
            self.stop_timer(step_start, "train/optimizer_step", sync=self.args.timing_breakdown)
            return losses.item()

        split_start = self.start_timer(sync=False)
        gt_pair_matchings = self.split_batch_pair_matchings(batch, gt_mapping_label)
        self.stop_timer(split_start, "train/split_gt_matchings")

        diffuse_start = self.start_timer(sync=False)
        diffused_pair_matchings = self.sample_partial_pair_matchings(gt_pair_matchings, t)
        self.stop_timer(diffuse_start, "train/sample_partial_matchings")

        edge_attr_start = self.start_timer(sync=False)
        diffused_mapping = self.pair_matchings_to_edge_attr(batch, diffused_pair_matchings)
        self.stop_timer(edge_attr_start, "train/pair_matchings_to_edge_attr")

        t = torch.from_numpy(t).float()
        forward_start = self.start_timer(sync=self.args.timing_breakdown)
        pred_mapping_label = self.model(batch, diffused_mapping.to(self.device), t.to(self.device))
        self.stop_timer(forward_start, "train/model_forward", sync=self.args.timing_breakdown)
        self.update_candidate_pruning_stats(self.unwrap_model().pop_last_topk_pruning_stats())

        loss_start = self.start_timer(sync=self.args.timing_breakdown)
        losses = mapping_loss(pred_mapping_label, batch)
        self.stop_timer(loss_start, "train/loss", sync=self.args.timing_breakdown)

        backward_start = self.start_timer(sync=self.args.timing_breakdown)
        losses.backward()
        self.stop_timer(backward_start, "train/backward", sync=self.args.timing_breakdown)

        step_start = self.start_timer(sync=self.args.timing_breakdown)
        self.optimizer.step()
        self.stop_timer(step_start, "train/optimizer_step", sync=self.args.timing_breakdown)
        return losses.item()
        
    def diffusion_ged_parallel(self,batch,test_k=100):
        if self.args.denoise_network in {"lightgt_disjoint", "lightgt_dense"}:
            return self.diffusion_ged_parallel_disjoint(batch, test_k=test_k)
        # generate k node matching matrices
        start_time = self.start_timer(sync=True)
        save_matching_artifacts = bool(getattr(self.args, "save_matching_artifacts", False))
        num_parallel_sampling = test_k
        data_list = batch.to_data_list()
        pair_sizes = batch.n.detach().cpu().tolist()
        eval_batch_size = len(data_list)
        expanded_data_list = []
        for data in data_list:
            expanded_data_list.extend([data for _ in range(num_parallel_sampling)])
        new_batch = Batch().from_data_list(expanded_data_list).to(self.device)
        self.stop_timer(start_time, "decode/expand_batch", sync=True)

        steps = self.args.inference_diffusion_steps
        forward_only_prefix_steps = max(int(getattr(self.args, "inference_forward_only_prefix_steps", 0)), 0)
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)
        pair_specs = []
        grouped_pair_indices = defaultdict(list)
        prepare_start = self.start_timer(sync=True)
        for pair_idx, (data, pair_size) in enumerate(zip(data_list, pair_sizes)):
            n1, n2 = pair_size
            block_start = pair_idx * num_parallel_sampling
            max_matching_size = min(n1, n2)
            edge_indices, flat_indices = self.build_block_dense_lookup(
                new_batch,
                block_start,
                num_parallel_sampling,
                n1,
                n2,
            )
            pair_specs.append(
                {
                    "block_start": block_start,
                    "n1": n1,
                    "n2": n2,
                    "max_matching_size": max_matching_size,
                    "edge_indices": edge_indices,
                    "flat_indices": flat_indices,
                    "base_score_cost": self.base_match_score_adjustment(data.to(self.device), n1, n2),
                }
            )
            grouped_pair_indices[(n1, n2, max_matching_size)].append(pair_idx)
        decode_groups = []
        for (n1, n2, max_matching_size), group_pair_indices in grouped_pair_indices.items():
            group_edge_indices = []
            group_flat_indices = []
            base_score_costs = []
            for group_offset, pair_idx in enumerate(group_pair_indices):
                pair_spec = pair_specs[pair_idx]
                group_edge_indices.append(pair_spec["edge_indices"])
                group_flat_indices.append(
                    pair_spec["flat_indices"] + group_offset * (num_parallel_sampling * n1 * n2)
                )
                base_score_costs.append(pair_spec["base_score_cost"])
            decode_groups.append(
                {
                    "pair_indices": group_pair_indices,
                    "group_size": len(group_pair_indices),
                    "total_batches": len(group_pair_indices) * num_parallel_sampling,
                    "n1": n1,
                    "n2": n2,
                    "max_matching_size": max_matching_size,
                    "edge_indices": torch.cat(group_edge_indices, dim=0),
                    "flat_indices": torch.cat(group_flat_indices, dim=0),
                    "base_score_costs": torch.stack(base_score_costs, dim=0),
                }
            )
        self.stop_timer(prepare_start, "decode/prepare_pair_states", sync=True)

        pack_start = self.start_timer(sync=True)
        mapping_t = new_batch.x.new_zeros((new_batch.edge_index_mapping.shape[1], 1))
        final_clean_matchings = [
            new_batch.x.new_zeros((num_parallel_sampling, pair_spec["n1"], pair_spec["n2"]))
            for pair_spec in pair_specs
        ]
        probability_trajectories = [[] for _ in pair_specs] if save_matching_artifacts else None
        self.stop_timer(pack_start, "decode/pack_partial_matching", sync=True)

        # reverse diffusion: predict a clean matching first, then re-noise to the next partial state
        for s in range(steps):
            t1,t2 = time_schedule(s)
            step_t = torch.full((eval_batch_size * num_parallel_sampling,), float(t1), device=self.device)
            forward_start = self.start_timer(sync=True)
            with torch.no_grad():
                pred_mapping_label = self.model(new_batch, mapping_t, step_t)
            self.update_candidate_pruning_stats(self.unwrap_model().pop_last_topk_pruning_stats())
            self.stop_timer(forward_start, "decode/reverse_step_forward", sync=True)

            if s < forward_only_prefix_steps:
                skipped_start = self.start_timer(sync=True)
                self.stop_timer(skipped_start, "decode/reverse_step_decode_and_renoise_skipped", sync=True)
                continue

            pred_probabilities = torch.sigmoid(pred_mapping_label)
            next_group_matchings = []
            clean_pair_matchings = [None] * len(pair_specs)

            decode_start = self.start_timer(sync=True)
            for group_spec in decode_groups:
                block_unpack_start = self.start_timer(sync=self.args.timing_breakdown)
                pred_pair_probabilities = self.grouped_edge_attr_to_dense(pred_probabilities, group_spec)
                pred_pair_probabilities = pred_pair_probabilities.view(
                    group_spec["group_size"],
                    num_parallel_sampling,
                    group_spec["n1"],
                    group_spec["n2"],
                )
                self.stop_timer(block_unpack_start, "decode/dense_prob_to_blocks", sync=self.args.timing_breakdown)
                target_size = int(round((1.0 - (float(t2) / float(self.diffusion.T))) * group_spec["max_matching_size"]))
                target_size = min(max(target_size, 0), group_spec["max_matching_size"])
                score_adjust_start = self.start_timer(sync=self.args.timing_breakdown)
                adjusted_scores = torch.logit(pred_pair_probabilities.clamp(1e-6, 1 - 1e-6))
                adjusted_scores = adjusted_scores - group_spec["base_score_costs"][:, None, :, :]
                adjusted_scores = adjusted_scores.view(-1, group_spec["n1"], group_spec["n2"])
                self.stop_timer(score_adjust_start, "decode/dense_score_adjustment", sync=self.args.timing_breakdown)
                if self.args.inference_decoder == "decoder_a":
                    clean_matchings, next_partial_matchings = self.batched_decoder_a_step(
                        adjusted_scores,
                        target_size,
                        t2,
                    )
                else:
                    greedy_decode_start = self.start_timer(sync=self.args.timing_breakdown)
                    clean_matchings = self.batched_constrained_matching_decode_by_mode(
                        adjusted_scores,
                        group_spec["max_matching_size"],
                    )
                    mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
                    if mode_now == "conf_row_greedy_n2":
                        greedy_key = "decode/greedy_full_decode_n2"
                    elif mode_now == "row_top1_unique_n2":
                        greedy_key = "decode/greedy_row_top1_unique_n2"
                    else:
                        greedy_key = "decode/greedy_full_decode"
                    self.stop_timer(greedy_decode_start, greedy_key, sync=self.args.timing_breakdown)
                    if t2 == 0:
                        next_partial_matchings = clean_matchings.clone()
                    else:
                        renoise_start = self.start_timer(sync=self.args.timing_breakdown)
                        next_partial_matchings = self.batched_sample_partial_from_clean_matching(
                            clean_matchings,
                            adjusted_scores,
                            target_size,
                        )
                        self.stop_timer(renoise_start, "decode/greedy_renoise", sync=self.args.timing_breakdown)
                clean_matchings = clean_matchings.view(
                    group_spec["group_size"],
                    num_parallel_sampling,
                    group_spec["n1"],
                    group_spec["n2"],
                )
                next_partial_matchings = next_partial_matchings.view(
                    group_spec["group_size"],
                    num_parallel_sampling,
                    group_spec["n1"],
                    group_spec["n2"],
                )
                collect_start = self.start_timer(sync=self.args.timing_breakdown)
                for group_offset, pair_idx in enumerate(group_spec["pair_indices"]):
                    clean_pair_matchings[pair_idx] = clean_matchings[group_offset]
                    if save_matching_artifacts:
                        probability_trajectories[pair_idx].append(
                            pred_pair_probabilities[group_offset].detach().to(device="cpu", dtype=torch.float16)
                        )
                next_group_matchings.append(next_partial_matchings)
                self.stop_timer(collect_start, "decode/dense_collect_group_outputs", sync=self.args.timing_breakdown)
            self.stop_timer(decode_start, "decode/reverse_step_decode_and_renoise", sync=True)

            final_clean_matchings = clean_pair_matchings
            repack_start = self.start_timer(sync=True)
            mapping_t = self.grouped_dense_blocks_to_edge_attr(
                new_batch.edge_index_mapping.shape[1],
                decode_groups,
                next_group_matchings,
                pred_probabilities,
            )
            self.stop_timer(repack_start, "decode/repack_partial_matching", sync=True)

        reduce_start = self.start_timer(sync=True)
        pair_probability_maps = self.split_batch_pair_matchings(new_batch, mapping_attr=pred_probabilities)
        lowprob_m = int(getattr(self.args, "lowprob_permute_m", 0))
        lowprob_max_cases = int(getattr(self.args, "lowprob_permute_max_cases", 0))
        reduced_pairs = []
        for pair_idx, (data, pair_spec) in enumerate(zip(data_list, pair_specs)):
            ged_eval_start = self.start_timer(sync=self.args.timing_breakdown)
            data_on_device = data.to(self.device)
            postprocess_time = 0.0
            min_ged, min_mapping = self.ged_from_clean_matchings(
                data_on_device,
                final_clean_matchings[pair_idx],
                pair_spec["n1"],
                pair_spec["n2"],
            )
            pre_swap_ged = min_ged.detach().clone() if torch.is_tensor(min_ged) else min_ged
            if lowprob_m > 0:
                seed_mapping = min_mapping[0] if min_mapping.dim() == 3 else min_mapping
                if seed_mapping.dim() == 2:
                    lowprob_start = self.start_timer(sync=self.args.timing_breakdown)
                    permuted_matchings = self.lowprob_permute_matchings(
                        seed_mapping.float(),
                        pair_probability_maps[pair_idx],
                        lowprob_m,
                        max_cases=lowprob_max_cases,
                    )
                    perm_ged, canonical_matchings = self.ged_values_from_clean_matchings(
                        data_on_device,
                        permuted_matchings,
                        pair_spec["n1"],
                        pair_spec["n2"],
                    )
                    best_idx = int(torch.argmin(perm_ged).item())
                    min_ged = perm_ged[best_idx]
                    min_mapping = canonical_matchings[best_idx:best_idx + 1]
                    postprocess_time += self.stop_timer(
                        lowprob_start,
                        "decode/lowprob_permute_local_search",
                        sync=self.args.timing_breakdown,
                    )
            if bool(getattr(self.args, "two_swap_local_search", False)):
                n1, n2 = pair_spec["n1"], pair_spec["n2"]
                if n1 == n2:
                    init_assignment = self._matching_to_assignment(min_mapping[0].bool())
                    if init_assignment is not None:
                        two_swap_start = self.start_timer(sync=self.args.timing_breakdown)
                        ls_matching, ls_ged, _, _ = self.batched_two_swap_ged_search(
                            data_on_device,
                            init_assignment.unsqueeze(0),
                            max_iter=int(getattr(self.args, "two_swap_max_iter", 50)),
                            eps=float(getattr(self.args, "two_swap_eps", 1e-9)),
                            chunk_size=(None if int(getattr(self.args, "two_swap_chunk_size", 0)) <= 0
                                        else int(getattr(self.args, "two_swap_chunk_size", 0))),
                        )
                        postprocess_time += self.stop_timer(
                            two_swap_start,
                            "decode/two_swap_local_search",
                            sync=self.args.timing_breakdown,
                        )
                        min_ged = ls_ged[0]
                        min_mapping = self._assignments_to_clean_matchings(ls_matching, n1, n2)
            self.stop_timer(ged_eval_start, "decode/final_ged_eval", sync=self.args.timing_breakdown)
            reduced_pairs.append((min_ged, min_mapping, postprocess_time, pre_swap_ged))
        self.stop_timer(reduce_start, "decode/final_ged_reduction", sync=True)
        elapsed = self.stop_timer(start_time, "decode/total", sync=True)
        per_pair_elapsed = elapsed / max(eval_batch_size, 1)
        results = []
        for pair_idx, (min_ged, min_mapping, postprocess_time, pre_swap_ged) in enumerate(reduced_pairs):
            extra_metrics = {"postprocess_time": float(postprocess_time), "pre_swap_ged": pre_swap_ged}
            if save_matching_artifacts:
                pair_artifact = {
                    "probability_maps": torch.stack(probability_trajectories[pair_idx], dim=0),
                    "greedy_matchings": final_clean_matchings[pair_idx].detach().to(device="cpu", dtype=torch.uint8),
                    "best_matching": min_mapping.detach().to(device="cpu", dtype=torch.uint8),
                }
            else:
                pair_artifact = None
            results.append((min_ged, min_mapping, per_pair_elapsed, pair_artifact, extra_metrics))

        if eval_batch_size == 1:
            return results[0]
        return results

    def diffusion_ged_parallel_disjoint(self, batch, test_k=100):
        start_time = self.start_timer(sync=True)
        save_matching_artifacts = bool(getattr(self.args, "save_matching_artifacts", False))
        num_parallel_sampling = test_k
        data_list = batch.to_data_list()
        pair_sizes = batch.n.detach().cpu().tolist()
        eval_batch_size = len(data_list)
        new_batch = self._expand_disjoint_batch_tensorized(batch, num_parallel_sampling)
        self.stop_timer(start_time, "decode/expand_batch", sync=True)

        steps = self.args.inference_diffusion_steps
        forward_only_prefix_steps = max(int(getattr(self.args, "inference_forward_only_prefix_steps", 0)), 0)
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)
        mapping_t = new_batch.x.new_zeros((new_batch.edge_index_mapping.shape[1], 1))
        final_selected_mask = torch.zeros(
            new_batch.edge_index_mapping.shape[1],
            dtype=torch.bool,
            device=self.device,
        )
        probability_trajectories = [[] for _ in range(eval_batch_size)] if save_matching_artifacts else None
        info_start = self.start_timer(sync=True)
        disjoint_model = self.unwrap_model()
        batch_info = disjoint_model.build_disjoint_batch_info(new_batch)
        self.stop_timer(info_start, "decode/build_disjoint_batch_info", sync=True)
        pair_owner = batch_info["edge_pair_id"] // num_parallel_sampling
        max_matching_size = torch.minimum(new_batch.n[:, 0], new_batch.n[:, 1]).long()
        decode_mode = self.args.inference_decoder

        decode_groups = None
        global_padded_decode = None
        if decode_mode == "constrained_greedy":
            pair_specs = []
            grouped_pair_indices = defaultdict(list)
            prepare_start = self.start_timer(sync=True)
            for pair_idx, (data, pair_size) in enumerate(zip(data_list, pair_sizes)):
                n1, n2 = pair_size
                block_start = pair_idx * num_parallel_sampling
                max_pair_matching_size = min(n1, n2)
                edge_indices, flat_indices = self.build_block_dense_lookup(
                    new_batch,
                    block_start,
                    num_parallel_sampling,
                    n1,
                    n2,
                )
                pair_specs.append(
                    {
                        "block_start": block_start,
                        "n1": n1,
                        "n2": n2,
                        "max_matching_size": max_pair_matching_size,
                        "edge_indices": edge_indices,
                        "flat_indices": flat_indices,
                        "base_score_cost": self.base_match_score_adjustment(data.to(self.device), n1, n2),
                    }
                )
                grouped_pair_indices[(n1, n2, max_pair_matching_size)].append(pair_idx)
            decode_groups = []
            for (n1, n2, max_pair_matching_size), group_pair_indices in grouped_pair_indices.items():
                group_edge_indices = []
                group_flat_indices = []
                base_score_costs = []
                for group_offset, pair_idx in enumerate(group_pair_indices):
                    pair_spec = pair_specs[pair_idx]
                    group_edge_indices.append(pair_spec["edge_indices"])
                    group_flat_indices.append(
                        pair_spec["flat_indices"] + group_offset * (num_parallel_sampling * n1 * n2)
                    )
                    base_score_costs.append(pair_spec["base_score_cost"])
                decode_groups.append(
                    {
                        "pair_indices": group_pair_indices,
                        "group_size": len(group_pair_indices),
                        "total_batches": len(group_pair_indices) * num_parallel_sampling,
                        "n1": n1,
                        "n2": n2,
                        "max_matching_size": max_pair_matching_size,
                        "edge_indices": torch.cat(group_edge_indices, dim=0),
                        "flat_indices": torch.cat(group_flat_indices, dim=0),
                        "base_score_costs": torch.stack(base_score_costs, dim=0),
                    }
                )
            global_padded_decode = self.build_global_padded_decode_spec(
                decode_groups=decode_groups,
                eval_batch_size=eval_batch_size,
                num_parallel_sampling=num_parallel_sampling,
                device=self.device,
            )
            self.stop_timer(prepare_start, "decode/prepare_pair_states", sync=True)
        elif decode_mode != "disjoint_uncertainty":
            raise ValueError(
                f"Unsupported decoder for lightgt_disjoint: {decode_mode}. "
                "Use constrained_greedy or disjoint_uncertainty."
            )

        for s in range(steps):
            t1, t2 = time_schedule(s)
            step_t = torch.full((eval_batch_size * num_parallel_sampling,), float(t1), device=self.device)
            forward_start = self.start_timer(sync=True)
            with torch.no_grad():
                pred_mapping_label = self.model(new_batch, mapping_t, step_t)
            self.update_candidate_pruning_stats(self.unwrap_model().pop_last_topk_pruning_stats())
            self.stop_timer(forward_start, "decode/reverse_step_forward", sync=True)

            if s < forward_only_prefix_steps:
                skipped_start = self.start_timer(sync=True)
                self.stop_timer(skipped_start, "decode/reverse_step_decode_and_renoise_skipped", sync=True)
                if self.args.log_step_ged_curve:
                    step_ged_start = self.start_timer(sync=self.args.timing_breakdown)
                    step_selected_mask = mapping_t[:, 0] > 0.5
                    step_sample_ged = disjoint_model.induced_ged_from_selected_edges(new_batch, step_selected_mask)
                    step_ged_matrix = step_sample_ged.view(eval_batch_size, num_parallel_sampling)
                    step_best_values = torch.min(step_ged_matrix, dim=1).values
                    self.update_step_ged_stats(s, step_best_values)
                    self.stop_timer(step_ged_start, "decode/step_ged_eval", sync=self.args.timing_breakdown)
                continue

            pred_probabilities = torch.sigmoid(pred_mapping_label).view(-1)
            target_size_start = self.start_timer(sync=self.args.timing_breakdown)
            if float(t2) == 0.0:
                target_size = max_matching_size
            else:
                ratio = 1.0 - (float(t2) / float(self.diffusion.T))
                target_size = torch.clamp(
                    torch.round(ratio * max_matching_size.float()).long(),
                    min=0,
                )
            self.stop_timer(target_size_start, "decode/disjoint_target_size", sync=self.args.timing_breakdown)

            decode_start = self.start_timer(sync=True)
            if decode_mode == "constrained_greedy":
                global_spec = global_padded_decode["group_spec"]
                pair_valid_mask = global_padded_decode["pair_valid_mask"]
                pair_max_sizes = global_padded_decode["pair_max_matching_size"]
                pad_n1 = global_padded_decode["pad_n1"]
                pad_n2 = global_padded_decode["pad_n2"]

                block_unpack_start = self.start_timer(sync=self.args.timing_breakdown)
                pred_pair_probabilities = self.grouped_edge_attr_to_dense(pred_mapping_label, global_spec)
                pred_pair_probabilities = torch.sigmoid(pred_pair_probabilities.view(
                    eval_batch_size,
                    num_parallel_sampling,
                    pad_n1,
                    pad_n2,
                ))
                valid4d = pair_valid_mask[:, None, :, :]
                pred_pair_probabilities = pred_pair_probabilities.masked_fill(~valid4d, 0.0)
                self.stop_timer(block_unpack_start, "decode/dense_prob_to_blocks", sync=self.args.timing_breakdown)

                score_adjust_start = self.start_timer(sync=self.args.timing_breakdown)
                adjusted_scores_4d = torch.logit(pred_pair_probabilities.clamp(1e-6, 1 - 1e-6))
                adjusted_scores_4d = adjusted_scores_4d - global_spec["base_score_costs"][:, None, :, :]
                adjusted_scores_4d = adjusted_scores_4d.masked_fill(~valid4d, float("-inf"))
                adjusted_scores = adjusted_scores_4d.view(-1, pad_n1, pad_n2)
                self.stop_timer(score_adjust_start, "decode/dense_score_adjustment", sync=self.args.timing_breakdown)

                greedy_decode_start = self.start_timer(sync=self.args.timing_breakdown)
                mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
                if mode_now == "global_n3":
                    full_sizes = pair_max_sizes[:, None].expand(-1, num_parallel_sampling).reshape(-1)
                    clean_matchings = self.batched_constrained_matching_decode_variable_size(
                        adjusted_scores,
                        full_sizes,
                    )
                else:
                    clean_matchings = self.batched_constrained_matching_decode_by_mode(
                        adjusted_scores,
                        int(pair_max_sizes.max().item()) if pair_max_sizes.numel() > 0 else 0,
                    )
                if mode_now == "conf_row_greedy_n2":
                    greedy_key = "decode/greedy_full_decode_n2"
                elif mode_now == "row_top1_unique_n2":
                    greedy_key = "decode/greedy_row_top1_unique_n2"
                else:
                    greedy_key = "decode/greedy_full_decode"
                self.stop_timer(greedy_decode_start, greedy_key, sync=self.args.timing_breakdown)

                clean_matchings = clean_matchings.view(eval_batch_size, num_parallel_sampling, pad_n1, pad_n2)
                clean_matchings = clean_matchings.masked_fill(~valid4d, 0.0)

                if t2 == 0:
                    next_partial_matchings = clean_matchings.clone()
                else:
                    renoise_start = self.start_timer(sync=self.args.timing_breakdown)
                    next_partial_matchings = torch.zeros_like(clean_matchings)
                    for pair_idx in range(eval_batch_size):
                        target_size_scalar = int(round((1.0 - (float(t2) / float(self.diffusion.T))) * int(pair_max_sizes[pair_idx].item())))
                        target_size_scalar = min(max(target_size_scalar, 0), int(pair_max_sizes[pair_idx].item()))
                        if target_size_scalar <= 0:
                            continue
                        next_partial_matchings[pair_idx] = self.batched_sample_partial_from_clean_matching(
                            clean_matchings[pair_idx],
                            adjusted_scores_4d[pair_idx],
                            target_size_scalar,
                        )
                    next_partial_matchings = next_partial_matchings.masked_fill(~valid4d, 0.0)
                    self.stop_timer(renoise_start, "decode/greedy_renoise", sync=self.args.timing_breakdown)

                if save_matching_artifacts:
                    collect_start = self.start_timer(sync=self.args.timing_breakdown)
                    for pair_idx, (n1, n2) in enumerate(pair_sizes):
                        probability_trajectories[pair_idx].append(
                            pred_pair_probabilities[pair_idx, :, :n1, :n2].detach().to(device="cpu", dtype=torch.float16)
                        )
                    self.stop_timer(collect_start, "decode/dense_collect_group_outputs", sync=self.args.timing_breakdown)

                repack_start = self.start_timer(sync=self.args.timing_breakdown)
                mapping_t = self.grouped_dense_blocks_to_edge_attr(
                    new_batch.edge_index_mapping.shape[1],
                    [global_spec],
                    [next_partial_matchings.view(-1, pad_n1, pad_n2)],
                    pred_mapping_label,
                )
                final_selected_mask = mapping_t[:, 0] > 0.5
                self.stop_timer(repack_start, "decode/repack_partial_matching", sync=self.args.timing_breakdown)
            else:
                match_start = self.start_timer(sync=self.args.timing_breakdown)
                matched_edges = disjoint_model.batched_uncertainty_matching(
                    new_batch,
                    pred_probabilities,
                    target_size=target_size,
                    lam=float(getattr(self.args, "disjoint_decoder_lam", 1.0)),
                    tau=getattr(self.args, "disjoint_decoder_tau", None),
                    max_iter=int(max_matching_size.max().item()) if max_matching_size.numel() > 0 else 0,
                )
                self.stop_timer(match_start, "decode/disjoint_match", sync=self.args.timing_breakdown)

                edge_attr_update_start = self.start_timer(sync=self.args.timing_breakdown)
                mapping_t = disjoint_model.edge_attr_from_matched_edges(
                    new_batch.edge_index_mapping.shape[1],
                    matched_edges,
                    pred_mapping_label,
                )
                final_selected_mask = mapping_t[:, 0] > 0.5
                self.stop_timer(edge_attr_update_start, "decode/disjoint_edge_attr_update", sync=self.args.timing_breakdown)
            self.stop_timer(decode_start, "decode/reverse_step_decode_and_renoise", sync=True)

            if self.args.log_step_ged_curve:
                step_ged_start = self.start_timer(sync=self.args.timing_breakdown)
                step_sample_ged = disjoint_model.induced_ged_from_selected_edges(new_batch, final_selected_mask)
                step_ged_matrix = step_sample_ged.view(eval_batch_size, num_parallel_sampling)
                step_best_values = torch.min(step_ged_matrix, dim=1).values
                self.update_step_ged_stats(s, step_best_values)
                self.stop_timer(step_ged_start, "decode/step_ged_eval", sync=self.args.timing_breakdown)

            if save_matching_artifacts and decode_mode == "disjoint_uncertainty":
                collect_start = self.start_timer(sync=self.args.timing_breakdown)
                for pair_idx in range(eval_batch_size):
                    pair_mask = pair_owner == pair_idx
                    probability_trajectories[pair_idx].append(
                        pred_probabilities[pair_mask].detach().to(device="cpu", dtype=torch.float16)
                    )
                self.stop_timer(collect_start, "decode/disjoint_collect_prob_maps", sync=self.args.timing_breakdown)

        reduce_start = self.start_timer(sync=True)
        sample_ged_start = self.start_timer(sync=self.args.timing_breakdown)
        sample_ged = disjoint_model.induced_ged_from_selected_edges(new_batch, final_selected_mask)
        self.stop_timer(sample_ged_start, "decode/final_sample_ged", sync=self.args.timing_breakdown)
        select_start = self.start_timer(sync=self.args.timing_breakdown)
        sample_ged_matrix = sample_ged.view(eval_batch_size, num_parallel_sampling)
        best_offsets = torch.argmin(sample_ged_matrix, dim=1)
        base_pair_ids = torch.arange(eval_batch_size, device=self.device) * num_parallel_sampling
        best_pair_ids = base_pair_ids + best_offsets
        best_values = sample_ged_matrix.gather(1, best_offsets.unsqueeze(1)).squeeze(1)
        self.stop_timer(select_start, "decode/final_best_sample_select", sync=self.args.timing_breakdown)
        extract_start = self.start_timer(sync=self.args.timing_breakdown)
        pair_probability_maps = disjoint_model.sparse_probabilities_for_selected_pairs(
            new_batch, best_pair_ids, pred_probabilities
        )
        lowprob_m = int(getattr(self.args, "lowprob_permute_m", 0))
        lowprob_max_cases = int(getattr(self.args, "lowprob_permute_max_cases", 0))
        best_matchings = disjoint_model.sparse_matchings_for_selected_pairs(
            new_batch, best_pair_ids, final_selected_mask
        )
        reduced_pairs = [(best_values[idx], best_matchings[idx], 0.0) for idx in range(len(best_matchings))]
        pre_swap_values = [value.detach().clone() if torch.is_tensor(value) else value for value in best_values]
        if lowprob_m > 0 or bool(getattr(self.args, "two_swap_local_search", False)):
            refined_pairs = []
            for pair_idx, (min_ged, min_mapping, _) in enumerate(reduced_pairs):
                n1, n2 = pair_sizes[pair_idx]
                data_on_device = data_list[pair_idx].to(self.device)
                postprocess_time = 0.0
                if lowprob_m > 0:
                    seed_mapping = min_mapping[0] if min_mapping.dim() == 3 else min_mapping
                    if seed_mapping.dim() == 2:
                        lowprob_start = self.start_timer(sync=self.args.timing_breakdown)
                        permuted_matchings = self.lowprob_permute_matchings(
                            seed_mapping.float(),
                            pair_probability_maps[pair_idx],
                            lowprob_m,
                            max_cases=lowprob_max_cases,
                        )
                        perm_ged, canonical_matchings = self.ged_values_from_clean_matchings(
                            data_on_device,
                            permuted_matchings,
                            n1,
                            n2,
                        )
                        best_idx = int(torch.argmin(perm_ged).item())
                        min_ged = perm_ged[best_idx]
                        min_mapping = canonical_matchings[best_idx]
                        self.stop_timer(
                            lowprob_start,
                            "decode/lowprob_permute_local_search",
                            sync=self.args.timing_breakdown,
                        )
                if not bool(getattr(self.args, "two_swap_local_search", False)):
                    refined_pairs.append((min_ged, min_mapping, postprocess_time))
                    continue
                if n1 != n2:
                    refined_pairs.append((min_ged, min_mapping, postprocess_time))
                    continue
                # Some decoders may return multiple candidate matchings [K, n1, n2].
                # Local 2-swap expects a single assignment seed, so pick the first candidate.
                seed_mapping = min_mapping
                if seed_mapping.dim() == 3:
                    if seed_mapping.shape[0] == 0:
                        refined_pairs.append((min_ged, min_mapping, postprocess_time))
                        continue
                    seed_mapping = seed_mapping[0]
                init_assignment = self._matching_to_assignment(seed_mapping.bool())
                if init_assignment is None:
                    refined_pairs.append((min_ged, min_mapping, postprocess_time))
                    continue
                two_swap_start = self.start_timer(sync=self.args.timing_breakdown)
                ls_matching, ls_ged, _, _ = self.batched_two_swap_ged_search(
                    data_on_device,
                    init_assignment.unsqueeze(0),
                    max_iter=int(getattr(self.args, "two_swap_max_iter", 50)),
                    eps=float(getattr(self.args, "two_swap_eps", 1e-9)),
                    chunk_size=(None if int(getattr(self.args, "two_swap_chunk_size", 0)) <= 0
                                else int(getattr(self.args, "two_swap_chunk_size", 0))),
                )
                postprocess_time += self.stop_timer(
                    two_swap_start,
                    "decode/two_swap_local_search",
                    sync=self.args.timing_breakdown,
                )
                refined_pairs.append((ls_ged[0], self._assignments_to_clean_matchings(ls_matching, n1, n2)[0], postprocess_time))
            reduced_pairs = refined_pairs
        self.stop_timer(extract_start, "decode/final_matching_extract", sync=self.args.timing_breakdown)
        self.stop_timer(reduce_start, "decode/final_ged_reduction", sync=True)

        elapsed = self.stop_timer(start_time, "decode/total", sync=True)
        per_pair_elapsed = elapsed / max(eval_batch_size, 1)
        results = []
        for pair_idx, (min_ged, min_mapping, postprocess_time) in enumerate(reduced_pairs):
            extra_metrics = {"pre_swap_ged": pre_swap_values[pair_idx], "postprocess_time": float(postprocess_time)}
            if save_matching_artifacts:
                pair_artifact = {
                    "probability_maps": probability_trajectories[pair_idx],
                    "greedy_matchings": min_mapping.detach().to(device="cpu", dtype=torch.uint8),
                    "best_matching": min_mapping.detach().to(device="cpu", dtype=torch.uint8),
                }
            else:
                pair_artifact = None
            results.append((min_ged, min_mapping, per_pair_elapsed, pair_artifact, extra_metrics))

        if eval_batch_size == 1:
            return results[0]
        return results
    
    def save(self, epoch):
        if not self.is_main_process:
            return
        output_path = self._checkpoint_path(epoch)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.unwrap_model().state_dict(), output_path)

    def load(self, epoch):
        state_dict = torch.load(self._checkpoint_path(epoch), map_location=self.device)
        strict = self.args.denoise_network != "lightgt_dense"
        load_result = self.unwrap_model().load_state_dict(state_dict, strict=strict)
        if (not strict) and self.is_main_process:
            missing = len(getattr(load_result, "missing_keys", []))
            unexpected = len(getattr(load_result, "unexpected_keys", []))
            print(
                f"[Load][lightgt_dense] non-strict load: missing_keys={missing}, unexpected_keys={unexpected}"
            )

    def _resolve_output_dir(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return Path(self.args.abs_path) / path

    def _checkpoint_path(self, epoch):
        model_dir = self._resolve_output_dir(self.args.model_path)
        return model_dir / f"{self.args.dataset}_{epoch}_{self.args.model_name}.pt"

    def _result_path(self, filename):
        result_dir = self._resolve_output_dir(self.args.result_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir / filename

    def _pair_result_shard_path(self, dataset, testing_graph_set, top_k_approach, test_k, rank):
        shard_name = (
            f"result_DiffGED_{dataset}_{testing_graph_set}_{top_k_approach}_{test_k}"
            f".pair_results.rank{rank}.json"
        )
        return self._result_path(shard_name)

    def _matching_artifact_shard_path(self, dataset, testing_graph_set, top_k_approach, test_k, rank):
        shard_name = (
            f"result_DiffGED_{dataset}_{testing_graph_set}_{top_k_approach}_{test_k}"
            f".matching_artifacts.rank{rank}.pt"
        )
        return self._result_path(shard_name)
    
    @staticmethod
    def cal_pk(num, pre, gt):
        num = min(num, len(pre))
        if num == 0:
            return 0
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num
    
    def score(self,testing_graph_set='test', test_k=100, top_k_approach='parallel'):
        assert test_k > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
    
        if self.is_main_process:
            print("\n\nEvalute DiffGED with {} topk {} on {} set.\n".format(top_k_approach,test_k,testing_graph_set))
        self.model.eval()
        self.reset_timing_stats()
        self.reset_memory_stats()
        self.reset_step_ged_stats()
        if self.module_timing_profiler is not None:
            self.module_timing_profiler.reset()
        score_start = self.start_timer(sync=self.args.timing_breakdown)
        pair_results_via_shards = self.has_precomputed_pairs and self.args.save_pair_results
        save_matching_artifacts = bool(getattr(self.args, "save_matching_artifacts", False))
        lightweight_summary = (
            self.has_precomputed_pairs
            and self.args.dataset in {
            "ogbg-code2",
            "ogbg-molhiv",
            "ogbg-molpcba",
            }
            and not pair_results_via_shards
            and not save_matching_artifacts
        )
        local_pred_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_gt_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_abs_error_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_pred_lt_gt_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_count = torch.zeros((), device=self.device, dtype=torch.float32)
        local_time_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_pre_swap_pred_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_pre_swap_abs_error_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_pre_swap_pred_lt_gt_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_pre_swap_count = torch.zeros((), device=self.device, dtype=torch.float32)
        local_two_swap_refined_better_than_pre_count = torch.zeros((), device=self.device, dtype=torch.float32)
        local_postprocess_time_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        local_chunks = []
        local_matching_artifacts = []
        iterator = tqdm(
            loader,
            file=sys.stdout,
            position=self.args.tqdm_position,
            dynamic_ncols=True,
        ) if self.show_progress else loader
        global_pair_limit = int(getattr(self.args, "max_test_pairs", 0))
        local_pair_budget = 0
        if global_pair_limit > 0:
            if self.distributed:
                local_pair_budget = max(
                    (global_pair_limit - self.rank + self.world_size - 1) // self.world_size,
                    0,
                )
            else:
                local_pair_budget = global_pair_limit
        local_pairs_processed = 0

        for batch_idx, batch in enumerate(iterator):
            if self.args.max_test_batches > 0 and batch_idx >= self.args.max_test_batches:
                break
            if global_pair_limit > 0 and local_pairs_processed >= local_pair_budget:
                break
            if global_pair_limit > 0:
                remaining_local = local_pair_budget - local_pairs_processed
                if remaining_local <= 0:
                    break
                batch_pairs = int(batch.ged.view(-1).numel())
                if batch_pairs > remaining_local:
                    data_list = batch.to_data_list()[:remaining_local]
                    batch = Batch.from_data_list(data_list)
            transfer_start = self.start_timer(sync=self.args.timing_breakdown)
            batch = batch.to(self.device)
            self.stop_timer(transfer_start, "score/batch_to_device", sync=self.args.timing_breakdown)
            model_out = self.diffusion_ged_parallel(batch,test_k)
            post_start = self.start_timer(sync=self.args.timing_breakdown)
            batch_outputs = self.normalize_batch_outputs(model_out)
            pred_ged, running_time = self.stack_batch_predictions(batch_outputs, batch.ged.device)
            self.stop_timer(post_start, "score/postprocess_predictions", sync=self.args.timing_breakdown)
            gt_values = batch.ged.view(-1)
            local_pairs_processed += int(gt_values.numel())
            local_pred_sum += pred_ged.float().sum()
            local_gt_sum += gt_values.float().sum()
            local_abs_error_sum += torch.abs(pred_ged - gt_values).float().sum()
            local_pred_lt_gt_sum += (pred_ged < gt_values).float().sum()
            local_count += float(gt_values.numel())
            local_time_sum += running_time.float().sum()
            for pair_idx, pair_out in enumerate(batch_outputs):
                pre_swap_ged = self.extract_pre_swap_ged(pair_out)
                if pre_swap_ged is None:
                    continue
                pair_postprocess_time = self.extract_postprocess_time(pair_out)
                if pair_postprocess_time is not None:
                    local_postprocess_time_sum += float(pair_postprocess_time)
                gt_scalar = float(gt_values[pair_idx].item())
                refined_ged = float(self.to_python_scalar(pair_out[0]))
                local_pre_swap_pred_sum += pre_swap_ged
                local_pre_swap_abs_error_sum += abs(pre_swap_ged - gt_scalar)
                local_pre_swap_pred_lt_gt_sum += float(pre_swap_ged < gt_scalar)
                local_pre_swap_count += 1.0
                local_two_swap_refined_better_than_pre_count += float(refined_ged < pre_swap_ged)

            if not lightweight_summary:
                pair_ids = batch.i_j.view(-1, 2).detach().cpu()
                gt_values_cpu = gt_values.detach().cpu()
                pair_results = []
                for pair_idx, pair_out in enumerate(batch_outputs):
                    pred_value = float(self.to_python_scalar(pair_out[0]))
                    candidate_matchings = pair_out[1].detach().cpu()
                    pair_results.append(
                        {
                            "pair": [int(pair_ids[pair_idx, 0].item()), int(pair_ids[pair_idx, 1].item())],
                            "pair_gid": [
                                int(self.gid[int(pair_ids[pair_idx, 0].item())]),
                            int(self.gid[int(pair_ids[pair_idx, 1].item())]),
                        ],
                        "pred_ged": pred_value,
                        "gt_ged": float(gt_values_cpu[pair_idx].item()),
                        "matching": self.matching_tensor_to_edge_list(candidate_matchings),
                        "matchings": self.serialize_matchings(candidate_matchings),
                        "num_best_matchings": int(candidate_matchings.shape[0]) if candidate_matchings.dim() == 3 else 1,
                    }
                )
                    if save_matching_artifacts and len(pair_out) > 3 and pair_out[3] is not None:
                        artifact = pair_out[3]
                        local_matching_artifacts.append(
                            {
                                "pair": [int(pair_ids[pair_idx, 0].item()), int(pair_ids[pair_idx, 1].item())],
                                "pair_gid": [
                                    int(self.gid[int(pair_ids[pair_idx, 0].item())]),
                                    int(self.gid[int(pair_ids[pair_idx, 1].item())]),
                                ],
                                "pred_ged": pred_value,
                                "gt_ged": float(gt_values_cpu[pair_idx].item()),
                                "probability_maps": artifact["probability_maps"],
                                "greedy_matchings": artifact["greedy_matchings"],
                                "best_matching": artifact["best_matching"],
                            }
                        )
                local_chunks.append(
                    {
                        "graph_id": pair_ids[:, 0],
                        "pred": pred_ged.detach().cpu(),
                        "gt": gt_values_cpu,
                        "time": running_time.detach().cpu(),
                        "pair_results": pair_results,
                    }
                )

        self.stop_timer(score_start, "score/total", sync=self.args.timing_breakdown)
        aggregated_timing = self.aggregate_timing_breakdown() if self.args.timing_breakdown else None
        aggregated_candidate_pruning = self.aggregate_candidate_pruning_stats()
        aggregated_step_ged = self.aggregate_step_ged_stats() if self.args.log_step_ged_curve else None
        aggregated_module_timing = self.aggregate_module_timing_breakdown() if self.module_timing_profiler is not None else None
        aggregated_memory = self.aggregate_memory_breakdown()
        summary_tensor = torch.stack(
            [
                local_pred_sum,
                local_gt_sum,
                local_abs_error_sum,
                local_pred_lt_gt_sum,
                local_count,
                local_time_sum,
                local_pre_swap_pred_sum,
                local_pre_swap_abs_error_sum,
                local_pre_swap_pred_lt_gt_sum,
                local_pre_swap_count,
                local_two_swap_refined_better_than_pre_count,
                local_postprocess_time_sum,
            ]
        )
        self._all_reduce_tensor(summary_tensor)
        if pair_results_via_shards:
            shard_pair_results = [pair_item for item in local_chunks for pair_item in item["pair_results"]]
            shard_output_path = self._pair_result_shard_path(
                self.args.dataset,
                testing_graph_set,
                top_k_approach,
                test_k,
                self.rank,
            )
            with open(shard_output_path, "w") as f:
                json.dump(shard_pair_results, f)
            self.barrier()

        if save_matching_artifacts:
            artifact_output_path = self._matching_artifact_shard_path(
                self.args.dataset,
                testing_graph_set,
                top_k_approach,
                test_k,
                self.rank,
            )
            torch.save(local_matching_artifacts, artifact_output_path)
            self.barrier()

        gathered_records = self._gather_objects(local_chunks) if (not lightweight_summary and not pair_results_via_shards) else None
        if self.is_main_process:
            (
                pred_sum,
                gt_sum,
                abs_error_sum,
                pred_lt_gt_sum,
                count_sum,
                time_sum,
                pre_swap_pred_sum,
                pre_swap_abs_error_sum,
                pre_swap_pred_lt_gt_sum,
                pre_swap_count_sum,
                two_swap_refined_better_than_pre_count_sum,
                postprocess_time_sum,
            ) = summary_tensor.tolist()
            num = int(count_sum)
            if num == 0:
                print("No evaluation records were produced. Please increase --max-test-batches or check the test loader.")
                self.barrier()
                return
            avg_pred_ged = round(pred_sum / count_sum, 3)
            avg_gt_ged = round(gt_sum / count_sum, 3)
            mae = round(abs_error_sum / count_sum, 3)
            time_usage = round(time_sum / count_sum, 5)
            pred_lt_gt_count = int(pred_lt_gt_sum)
            pred_lt_gt_ratio = round(pred_lt_gt_sum / count_sum, 3)
            has_pre_swap = pre_swap_count_sum > 0
            if has_pre_swap:
                if postprocess_time_sum <= 0 and aggregated_timing is not None:
                    postprocess_time_sum = float(
                        self._timing_entry_to_seconds(
                            aggregated_timing.get("decode/lowprob_permute_local_search", 0.0)
                        )
                        + self._timing_entry_to_seconds(
                            aggregated_timing.get("decode/two_swap_local_search", 0.0)
                        )
                    )
                pre_swap_avg_pred_ged = round(pre_swap_pred_sum / pre_swap_count_sum, 3)
                pre_swap_mae = round(pre_swap_abs_error_sum / pre_swap_count_sum, 3)
                pre_swap_pred_lt_gt_count = int(pre_swap_pred_lt_gt_sum)
                pre_swap_pred_lt_gt_ratio = round(pre_swap_pred_lt_gt_sum / pre_swap_count_sum, 3)
                swap_delta_mae = round(pre_swap_mae - mae, 3)
                swap_delta_avg_pred_ged = round(pre_swap_avg_pred_ged - avg_pred_ged, 3)
                swap_delta_pred_lt_gt_count = pre_swap_pred_lt_gt_count - pred_lt_gt_count
                swap_delta_pred_lt_gt_ratio = round(pre_swap_pred_lt_gt_ratio - pred_lt_gt_ratio, 3)
                two_swap_refined_better_than_pre_count = int(two_swap_refined_better_than_pre_count_sum)
                two_swap_refined_better_than_pre_ratio = round(
                    two_swap_refined_better_than_pre_count_sum / pre_swap_count_sum, 3
                )
                postprocess_time_total = round(postprocess_time_sum, 5)
                postprocess_time_avg = round(postprocess_time_sum / pre_swap_count_sum, 5)
            output_result_path = self._result_path(
                f'result_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}_{test_k}.json'
            )
            if lightweight_summary:
                self.results.append((
                    'model_name', 'topk_approach', 'dataset', 'graph_set', '#testing_pairs',
                    'time_usage(s/p)', 'mae', 'avg_pred_ged', 'avg_gt_ged',
                    'pred_lt_gt_count', 'pred_lt_gt_ratio'
                ))
                self.results.append((
                    self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num,
                    time_usage, mae, avg_pred_ged, avg_gt_ged, pred_lt_gt_count, pred_lt_gt_ratio
                ))

                print(*self.results[-2], sep='\t')
                print(*self.results[-1], sep='\t')
                if has_pre_swap:
                    print(
                        "postprocess_pre_metrics\t"
                        f"mae={pre_swap_mae}\tavg_pred_ged={pre_swap_avg_pred_ged}\t"
                        f"pred_lt_gt_count={pre_swap_pred_lt_gt_count}\tpred_lt_gt_ratio={pre_swap_pred_lt_gt_ratio}"
                    )
                    print(
                        "postprocess_delta(pre-refined)\t"
                        f"mae={swap_delta_mae}\tavg_pred_ged={swap_delta_avg_pred_ged}\t"
                        f"pred_lt_gt_count={swap_delta_pred_lt_gt_count}\tpred_lt_gt_ratio={swap_delta_pred_lt_gt_ratio}\t"
                        f"refined_lt_pre_count={two_swap_refined_better_than_pre_count}\t"
                        f"refined_lt_pre_ratio={two_swap_refined_better_than_pre_ratio}\t"
                        f"postprocess_time_total(s)={postprocess_time_total}\t"
                        f"postprocess_time_avg(s/pair)={postprocess_time_avg}"
                    )
                self.print_timing_breakdown(num, aggregated=aggregated_timing)
                self.print_candidate_pruning_stats(aggregated=aggregated_candidate_pruning)
                self.print_memory_breakdown(aggregated=aggregated_memory)
                if aggregated_module_timing:
                    self.print_module_timing_breakdown(num, aggregated=aggregated_module_timing)
                    self.save_module_timing_artifacts(num, aggregated=aggregated_module_timing)

                with open(output_result_path, 'w') as f:
                    json.dump({
                        'time': time_usage,
                        'mae': mae,
                        'avg_pred_ged': avg_pred_ged,
                        'avg_gt_ged': avg_gt_ged,
                        'pred_lt_gt_count': pred_lt_gt_count,
                        'pred_lt_gt_ratio': pred_lt_gt_ratio,
                        'pre_swap_metrics': ({
                            'count': int(pre_swap_count_sum),
                            'mae': pre_swap_mae,
                            'avg_pred_ged': pre_swap_avg_pred_ged,
                            'pred_lt_gt_count': pre_swap_pred_lt_gt_count,
                            'pred_lt_gt_ratio': pre_swap_pred_lt_gt_ratio,
                        } if has_pre_swap else {}),
                        'swap_delta_pre_minus_refined': ({
                            'mae': swap_delta_mae,
                            'avg_pred_ged': swap_delta_avg_pred_ged,
                            'pred_lt_gt_count': swap_delta_pred_lt_gt_count,
                            'pred_lt_gt_ratio': swap_delta_pred_lt_gt_ratio,
                            'refined_lt_pre_count': two_swap_refined_better_than_pre_count,
                            'refined_lt_pre_ratio': two_swap_refined_better_than_pre_ratio,
                            'postprocess_time_total_s': postprocess_time_total,
                            'postprocess_time_avg_s_per_pair': postprocess_time_avg,
                        } if has_pre_swap else {}),
                        'timing_breakdown': dict(sorted(self.timing_totals.items())) if self.args.timing_breakdown else {},
                        'timing_counts': dict(sorted(self.timing_counts.items())) if self.args.timing_breakdown else {},
                        'timing_breakdown_distributed': dict(sorted(aggregated_timing.items())) if self.args.timing_breakdown and aggregated_timing is not None else {},
                        'candidate_pruning_summary': self.candidate_pruning_summary_payload(aggregated_candidate_pruning),
                        'step_ged_curve': self.step_ged_curve_payload(aggregated_step_ged) if self.args.log_step_ged_curve else {},
                        'memory_breakdown_distributed': aggregated_memory,
                    },f)
                self.barrier()
                return

            if pair_results_via_shards:
                pair_results = []
                for shard_rank in range(self.world_size if self.distributed else 1):
                    shard_path = self._pair_result_shard_path(
                        self.args.dataset,
                        testing_graph_set,
                        top_k_approach,
                        test_k,
                        shard_rank,
                    )
                    if shard_path.exists():
                        with open(shard_path, "r") as f:
                            pair_results.extend(json.load(f))
                pair_results.sort(key=lambda x: (x["pair"][0], x["pair"][1]))

                self.results.append((
                    'model_name', 'topk_approach', 'dataset', 'graph_set', '#testing_pairs',
                    'time_usage(s/p)', 'mae', 'avg_pred_ged', 'avg_gt_ged',
                    'pred_lt_gt_count', 'pred_lt_gt_ratio'
                ))
                self.results.append((
                    self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num,
                    time_usage, mae, avg_pred_ged, avg_gt_ged, pred_lt_gt_count, pred_lt_gt_ratio
                ))

                print(*self.results[-2], sep='\t')
                print(*self.results[-1], sep='\t')
                if has_pre_swap:
                    print(
                        "postprocess_pre_metrics\t"
                        f"mae={pre_swap_mae}\tavg_pred_ged={pre_swap_avg_pred_ged}\t"
                        f"pred_lt_gt_count={pre_swap_pred_lt_gt_count}\tpred_lt_gt_ratio={pre_swap_pred_lt_gt_ratio}"
                    )
                    print(
                        "postprocess_delta(pre-refined)\t"
                        f"mae={swap_delta_mae}\tavg_pred_ged={swap_delta_avg_pred_ged}\t"
                        f"pred_lt_gt_count={swap_delta_pred_lt_gt_count}\tpred_lt_gt_ratio={swap_delta_pred_lt_gt_ratio}\t"
                        f"refined_lt_pre_count={two_swap_refined_better_than_pre_count}\t"
                        f"refined_lt_pre_ratio={two_swap_refined_better_than_pre_ratio}"
                    )
                self.print_timing_breakdown(num, aggregated=aggregated_timing)
                self.print_candidate_pruning_stats(aggregated=aggregated_candidate_pruning)
                if aggregated_module_timing:
                    self.print_module_timing_breakdown(num, aggregated=aggregated_module_timing)
                    self.save_module_timing_artifacts(num, aggregated=aggregated_module_timing)

                with open(output_result_path, 'w') as f:
                    json.dump({
                        'time': time_usage,
                        'mae': mae,
                        'avg_pred_ged': avg_pred_ged,
                        'avg_gt_ged': avg_gt_ged,
                        'pred_lt_gt_count': pred_lt_gt_count,
                        'pred_lt_gt_ratio': pred_lt_gt_ratio,
                        'pre_swap_metrics': ({
                            'count': int(pre_swap_count_sum),
                            'mae': pre_swap_mae,
                            'avg_pred_ged': pre_swap_avg_pred_ged,
                            'pred_lt_gt_count': pre_swap_pred_lt_gt_count,
                            'pred_lt_gt_ratio': pre_swap_pred_lt_gt_ratio,
                        } if has_pre_swap else {}),
                        'swap_delta_pre_minus_refined': ({
                            'mae': swap_delta_mae,
                            'avg_pred_ged': swap_delta_avg_pred_ged,
                            'pred_lt_gt_count': swap_delta_pred_lt_gt_count,
                            'pred_lt_gt_ratio': swap_delta_pred_lt_gt_ratio,
                            'refined_lt_pre_count': two_swap_refined_better_than_pre_count,
                            'refined_lt_pre_ratio': two_swap_refined_better_than_pre_ratio,
                        } if has_pre_swap else {}),
                        'timing_breakdown': dict(sorted(self.timing_totals.items())) if self.args.timing_breakdown else {},
                        'timing_counts': dict(sorted(self.timing_counts.items())) if self.args.timing_breakdown else {},
                        'timing_breakdown_distributed': dict(sorted(aggregated_timing.items())) if self.args.timing_breakdown and aggregated_timing is not None else {},
                        'candidate_pruning_summary': self.candidate_pruning_summary_payload(aggregated_candidate_pruning),
                        'step_ged_curve': self.step_ged_curve_payload(aggregated_step_ged) if self.args.log_step_ged_curve else {},
                        'pair_results': pair_results,
                    }, f)
                self.barrier()
                return

            records = [item for rank_records in gathered_records for item in rank_records]
            pred = torch.cat([item["pred"] for item in records], dim=0)
            gt = torch.cat([item["gt"] for item in records], dim=0)
            pair_results = [pair_item for item in records for pair_item in item["pair_results"]]
            pair_results.sort(key=lambda x: (x["pair"][0], x["pair"][1]))
            num_acc = (pred == gt).sum().item()
            acc = round(num_acc / num, 3)

            if self.has_precomputed_pairs:
                self.results.append((
                    'model_name', 'topk_approach', 'dataset', 'graph_set', '#testing_pairs',
                    'time_usage(s/p)', 'mae', 'avg_pred_ged', 'avg_gt_ged',
                    'pred_lt_gt_count', 'pred_lt_gt_ratio'
                ))
                self.results.append((
                    self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num,
                    time_usage, mae, avg_pred_ged, avg_gt_ged, pred_lt_gt_count, pred_lt_gt_ratio
                ))

                print(*self.results[-2], sep='\t')
                print(*self.results[-1], sep='\t')
                self.print_timing_breakdown(num, aggregated=aggregated_timing)
                self.print_candidate_pruning_stats(aggregated=aggregated_candidate_pruning)
                if aggregated_module_timing:
                    self.print_module_timing_breakdown(num, aggregated=aggregated_module_timing)
                    self.save_module_timing_artifacts(num, aggregated=aggregated_module_timing)

                with open(output_result_path, 'w') as f:
                    json.dump({
                        'time': time_usage,
                        'mae': mae,
                        'acc': acc,
                        'avg_pred_ged': avg_pred_ged,
                        'avg_gt_ged': avg_gt_ged,
                        'pred_lt_gt_count': pred_lt_gt_count,
                        'pred_lt_gt_ratio': pred_lt_gt_ratio,
                        'timing_breakdown': dict(sorted(self.timing_totals.items())) if self.args.timing_breakdown else {},
                        'timing_counts': dict(sorted(self.timing_counts.items())) if self.args.timing_breakdown else {},
                        'timing_breakdown_distributed': dict(sorted(aggregated_timing.items())) if self.args.timing_breakdown and aggregated_timing is not None else {},
                        'candidate_pruning_summary': self.candidate_pruning_summary_payload(aggregated_candidate_pruning),
                        'step_ged_curve': self.step_ged_curve_payload(aggregated_step_ged) if self.args.log_step_ged_curve else {},
                        'pair_results': pair_results,
                    }, f)
                self.barrier()
                return

        if not self.is_main_process:
            self.barrier()
            return

        graph_id = torch.cat([item["graph_id"] for item in records], dim=0)
        num_fea = (pred >= gt).sum().item()
        rho = []
        tau = []
        pk10 = []
        pk20 = []
        pres = {}
        gts = {}
        for gid, pred_value, gt_value in zip(graph_id.tolist(), pred.tolist(), gt.tolist()):
            pres.setdefault(gid, []).append(pred_value)
            gts.setdefault(gid, []).append(gt_value)

        for graph_id in pres:
            rho.append(spearmanr(pres[graph_id],gts[graph_id])[0])
            tau.append(kendalltau(pres[graph_id],gts[graph_id])[0])
            pk10.append(self.cal_pk(10, pres[graph_id],gts[graph_id]))
            pk20.append(self.cal_pk(20, pres[graph_id],gts[graph_id]))

        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(('model_name', 'topk_approach', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/p)', 'mae', 'acc',
                            'avg_pred_ged', 'avg_gt_ged', 'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num, time_usage, mae, acc,
                            avg_pred_ged, avg_gt_ged, fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        if has_pre_swap:
            print(
                "postprocess_pre_metrics\t"
                f"mae={pre_swap_mae}\tavg_pred_ged={pre_swap_avg_pred_ged}\t"
                f"pred_lt_gt_count={pre_swap_pred_lt_gt_count}\tpred_lt_gt_ratio={pre_swap_pred_lt_gt_ratio}"
            )
            print(
                "postprocess_delta(pre-refined)\t"
                f"mae={swap_delta_mae}\tavg_pred_ged={swap_delta_avg_pred_ged}\t"
                f"pred_lt_gt_count={swap_delta_pred_lt_gt_count}\tpred_lt_gt_ratio={swap_delta_pred_lt_gt_ratio}\t"
                f"refined_lt_pre_count={two_swap_refined_better_than_pre_count}\t"
                f"refined_lt_pre_ratio={two_swap_refined_better_than_pre_ratio}\t"
                f"postprocess_time_total(s)={postprocess_time_total}\t"
                f"postprocess_time_avg(s/pair)={postprocess_time_avg}"
            )
        self.print_timing_breakdown(num, aggregated=aggregated_timing)
        self.print_candidate_pruning_stats(aggregated=aggregated_candidate_pruning)
        if aggregated_module_timing:
            self.print_module_timing_breakdown(num, aggregated=aggregated_module_timing)
            self.save_module_timing_artifacts(num, aggregated=aggregated_module_timing)

        with open(output_result_path, 'w') as f:
            json.dump({
                'time': time_usage,
                'mae': mae,
                'acc': acc,
                'avg_pred_ged': avg_pred_ged,
                'avg_gt_ged': avg_gt_ged,
                'fea': fea,
                'rho': rho,
                'tau': tau,
                'pk10': pk10,
                'pk20': pk20,
                'pre_swap_metrics': ({
                    'count': int(pre_swap_count_sum),
                    'mae': pre_swap_mae,
                    'avg_pred_ged': pre_swap_avg_pred_ged,
                    'pred_lt_gt_count': pre_swap_pred_lt_gt_count,
                    'pred_lt_gt_ratio': pre_swap_pred_lt_gt_ratio,
                } if has_pre_swap else {}),
                'swap_delta_pre_minus_refined': ({
                    'mae': swap_delta_mae,
                    'avg_pred_ged': swap_delta_avg_pred_ged,
                    'pred_lt_gt_count': swap_delta_pred_lt_gt_count,
                    'pred_lt_gt_ratio': swap_delta_pred_lt_gt_ratio,
                    'refined_lt_pre_count': two_swap_refined_better_than_pre_count,
                    'refined_lt_pre_ratio': two_swap_refined_better_than_pre_ratio,
                    'postprocess_time_total_s': postprocess_time_total,
                    'postprocess_time_avg_s_per_pair': postprocess_time_avg,
                } if has_pre_swap else {}),
                'timing_breakdown': dict(sorted(self.timing_totals.items())) if self.args.timing_breakdown else {},
                'timing_counts': dict(sorted(self.timing_counts.items())) if self.args.timing_breakdown else {},
                'timing_breakdown_distributed': dict(sorted(aggregated_timing.items())) if self.args.timing_breakdown and aggregated_timing is not None else {},
                'candidate_pruning_summary': self.candidate_pruning_summary_payload(aggregated_candidate_pruning),
                'step_ged_curve': self.step_ged_curve_payload(aggregated_step_ged) if self.args.log_step_ged_curve else {},
                'pair_results': pair_results,
            },f)
        self.barrier()
    
    def analyze_topk(self,testing_graph_set = 'test',k_range=[1,10,20,30,40,50,60,70,80,90,100],top_k_approach='parallel'):
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nAnalyze {} topk approach on {} set.\n".format(top_k_approach,testing_graph_set))
        self.model.eval()

        num = 0  # total testing number
        time_usage = {i:0 for i in k_range}
        
          # ged mae
        mae = {i:[] for i in k_range}
        num_acc = {i:0 for i in k_range}  
        acc = {}
       
        for k in k_range:
            num = 0
            time_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            abs_error_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            correct_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            progress_loader = tqdm(
                loader,
                file=sys.stdout,
                position=self.args.tqdm_position,
                dynamic_ncols=True,
            ) if self.show_progress else loader
            for batch in progress_loader:
                batch = batch.to(self.device)
                model_out = self.diffusion_ged_parallel(batch,k)
                batch_outputs = self.normalize_batch_outputs(model_out)
                pred_ged, running_time = self.stack_batch_predictions(batch_outputs, batch.ged.device)
                gt_values = batch.ged.view(-1)
                num += int(gt_values.numel())
                time_sum += running_time.sum()
                abs_error_sum += torch.abs(pred_ged - gt_values).sum()
                correct_sum += (pred_ged == gt_values).float().sum()
               
            time_usage[k] = round((time_sum / num).item(),5)
            mae[k] = round((abs_error_sum / num).item(),3)
            acc[k] = round((correct_sum / num).item(),3)
          
            print(f'dataset: {self.args.dataset}, topk_approach: {top_k_approach}, k: {k}, avg time: {time_usage[k]}, mae: {mae[k]}, acc: {acc[k]}')
            
        with open(
            self._result_path(
                f'topk_analysis_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}.json'
            ),
            'w',
        ) as f:
            json.dump({'time':time_usage,'mae':mae,'acc':acc},f)
    
    def analyze_solution_diversity(self,testing_graph_set = 'test',top_k_approach='parallel',test_k=100):
        assert test_k > 0
        if testing_graph_set == 'test':
            loader = self.testing_data_loader
        elif testing_graph_set == 'small':
            loader = self.testing_data_small_loader
        elif testing_graph_set == 'large':
            loader = self.testing_data_large_loader
        
        print("\n\nAnalyze Solution Diversity with {} topk {} on {} set.\n".format(top_k_approach,test_k,testing_graph_set))
        self.model.eval()

        diversity_pred_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        diversity_gt_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        num = 0
        
        progress_loader = tqdm(
            loader,
            file=sys.stdout,
            position=self.args.tqdm_position,
            dynamic_ncols=True,
        ) if self.show_progress else loader
        for batch in progress_loader:
            batch = batch.to(self.device)
            model_out = self.diffusion_ged_parallel(batch,test_k)
            batch_outputs = self.normalize_batch_outputs(model_out)
            pred_ged, _ = self.stack_batch_predictions(batch_outputs, batch.ged.device)
            gt_values = batch.ged.view(-1)
            distinct_counts = []
            for pair_idx, pair_out in enumerate(batch_outputs):
                pre_mappings = pair_out[1]
                distinct_counts.append(torch.unique(torch.argmax(pre_mappings,dim=-1),dim=0).shape[0])
            distinct_counts = torch.tensor(distinct_counts, device=self.device, dtype=torch.float32)
            diversity_pred_sum += distinct_counts.sum()
            diversity_gt_sum += (distinct_counts * (pred_ged == gt_values).float()).sum()
            num += int(gt_values.numel())

        diversity_pred_ged = (diversity_pred_sum / num).item()
        diversity_gt_ged = (diversity_gt_sum / num).item()
        print(f'dataset: {self.args.dataset}, topk_approach: {top_k_approach}, k: {test_k}, diversity_pred_ged: {diversity_pred_ged}, diversity_gt_ged: {diversity_gt_ged}')

        with open(
            self._result_path(
                f'diversity_analysis_DiffGED_{self.args.dataset}_{testing_graph_set}_{top_k_approach}_{test_k}.json'
            ),
            'w',
        ) as f:
            json.dump({'diversity_pred_ged':diversity_pred_ged,' diversity_gt_ged':diversity_gt_ged},f)

    @staticmethod
    def _safe_torch_load(path):
        try:
            return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _matching_to_2d_tensor(matching_tensor):
        if matching_tensor.dim() == 3:
            return matching_tensor[0].bool()
        return matching_tensor.bool()

    @staticmethod
    def _row_to_col_assignment(matching_2d):
        assignment = {}
        for row_idx in range(matching_2d.shape[0]):
            cols = torch.where(matching_2d[row_idx])[0]
            if cols.numel() > 0:
                assignment[row_idx] = int(cols[0].item())
        return assignment

    def _resolve_artifact_paths(self):
        artifact_path = str(getattr(self.args, "artifact_path", "") or "").strip()
        if not artifact_path:
            raise ValueError("--artifact-path is required for experiment=artifact_local_search.")

        if any(token in artifact_path for token in ["*", "?", "["]):
            matches = [Path(path) for path in sorted(glob.glob(artifact_path))]
        else:
            candidate = Path(artifact_path)
            if candidate.is_dir():
                matches = sorted(candidate.glob("*.matching_artifacts.rank*.pt"))
            elif candidate.exists():
                matches = [candidate]
            else:
                matches = []

        if not matches:
            raise FileNotFoundError("No matching artifact files found for '{}'.".format(artifact_path))
        return matches

    def _default_local_search_output_prefix(self, artifact_paths):
        explicit = str(getattr(self.args, "local_search_output_prefix", "") or "").strip()
        if explicit:
            return Path(explicit)
        if len(artifact_paths) == 1 and artifact_paths[0].is_file():
            return artifact_paths[0].parent / (artifact_paths[0].stem + ".local_search")
        return artifact_paths[0].parent / (
            f"artifact_local_search_release{int(self.args.local_search_release_count)}"
            f"_extra{int(self.args.local_search_extra_cols)}"
        )

    def _select_best_matching_probability_map(self, artifact_item):
        best_matching = self._matching_to_2d_tensor(artifact_item["best_matching"])
        greedy_matchings = artifact_item["greedy_matchings"].bool()
        exact_match = (
            greedy_matchings.reshape(greedy_matchings.shape[0], -1)
            == best_matching.reshape(1, -1)
        ).all(dim=1)
        if exact_match.any():
            chosen_idx = int(torch.where(exact_match)[0][0].item())
        else:
            overlap = (
                greedy_matchings & best_matching.unsqueeze(0)
            ).reshape(greedy_matchings.shape[0], -1).sum(dim=1)
            chosen_idx = int(torch.argmax(overlap).item())

        probability_maps = artifact_item["probability_maps"]
        step_idx = int(self.args.local_search_probability_step)
        if step_idx < 0:
            step_idx += int(probability_maps.shape[0])
        step_idx = min(max(step_idx, 0), int(probability_maps.shape[0]) - 1)
        return best_matching, probability_maps[step_idx, chosen_idx].float(), chosen_idx, step_idx

    def _enumerate_local_matching_variants(self, base_matching, probability_map):
        row_assignment = self._row_to_col_assignment(base_matching)
        matched_items = [
            {
                "row": int(row_idx),
                "col": int(col_idx),
                "confidence": float(probability_map[row_idx, col_idx].item()),
            }
            for row_idx, col_idx in row_assignment.items()
        ]
        matched_items.sort(key=lambda item: (item["confidence"], item["row"], item["col"]))

        release_count = min(int(self.args.local_search_release_count), len(matched_items))
        released = matched_items[:release_count]
        released_rows = [item["row"] for item in released]
        released_cols = [item["col"] for item in released]

        col_has_match = base_matching.any(dim=0)
        unmatched_cols = torch.where(~col_has_match)[0].tolist()
        extra_col_limit = max(int(self.args.local_search_extra_cols), 0)
        scored_unmatched_cols = []
        if released_rows and unmatched_cols:
            released_probabilities = probability_map[torch.tensor(released_rows, dtype=torch.long)]
            for col_idx in unmatched_cols:
                col_scores = released_probabilities[:, col_idx]
                scored_unmatched_cols.append(
                    (
                        float(torch.max(col_scores).item()),
                        float(torch.mean(col_scores).item()),
                        int(col_idx),
                    )
                )
            scored_unmatched_cols.sort(key=lambda item: (-item[0], -item[1], item[2]))
        extra_cols = [item[2] for item in scored_unmatched_cols[:extra_col_limit]]
        candidate_cols = released_cols + extra_cols

        fixed_matching = base_matching.clone()
        if released_rows:
            fixed_matching[torch.tensor(released_rows, dtype=torch.long)] = False

        candidate_matchings = []
        candidate_records = []
        for assigned_cols in itertools.permutations(candidate_cols, len(released_rows)):
            candidate = fixed_matching.clone()
            assignment_pairs = []
            for row_idx, col_idx in zip(released_rows, assigned_cols):
                candidate[row_idx, col_idx] = True
                assignment_pairs.append([int(row_idx), int(col_idx)])
            candidate_matchings.append(candidate)
            candidate_records.append({"assignment": assignment_pairs})

        return {
            "released_matches": released,
            "released_rows": released_rows,
            "released_cols": released_cols,
            "extra_cols": extra_cols,
            "candidate_cols": candidate_cols,
            "candidate_matchings": candidate_matchings,
            "candidate_records": candidate_records,
        }

    def analyze_matching_artifacts_local_search(self):
        artifact_paths = self._resolve_artifact_paths()
        output_prefix = self._default_local_search_output_prefix(artifact_paths)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        pairs_output_path = output_prefix.with_suffix(".pairs.jsonl")
        summary_output_path = output_prefix.with_suffix(".summary.json")

        max_pairs = max(int(self.args.local_search_max_pairs), 0)
        processed_pairs = 0
        improved_pairs = 0
        equal_pairs = 0
        skipped_pairs = 0
        improvement_histogram = defaultdict(int)
        per_shard_stats = []

        with open(pairs_output_path, "w") as pair_writer:
            for artifact_path in artifact_paths:
                shard_start = time.perf_counter()
                shard_records = self._safe_torch_load(artifact_path)
                shard_total = len(shard_records)
                shard_processed = 0
                shard_improved = 0
                if self.is_main_process:
                    print(f"[ArtifactLocalSearch] loaded {artifact_path} with {shard_total} pairs", flush=True)

                for artifact_item in shard_records:
                    if max_pairs > 0 and processed_pairs >= max_pairs:
                        break
                    gid_1, gid_2 = [int(value) for value in artifact_item["pair_gid"]]
                    idx_1 = self.gid_to_index.get(gid_1)
                    idx_2 = self.gid_to_index.get(gid_2)
                    if idx_1 is None or idx_2 is None:
                        skipped_pairs += 1
                        continue

                    pair_data = self.pack_graph_pair((0, idx_1, idx_2))
                    n1, n2 = [int(value) for value in pair_data.n.view(-1).tolist()]
                    best_matching, probability_map, chosen_sample_idx, chosen_step_idx = self._select_best_matching_probability_map(artifact_item)
                    local_variants = self._enumerate_local_matching_variants(best_matching, probability_map)
                    if not local_variants["candidate_matchings"]:
                        skipped_pairs += 1
                        continue

                    pair_data_device = pair_data.to(self.device)
                    baseline_ged_tensor, _ = self.ged_values_from_clean_matchings(
                        pair_data_device,
                        best_matching.unsqueeze(0).float().to(self.device),
                        n1,
                        n2,
                    )
                    candidate_geds, _ = self.ged_values_from_clean_matchings(
                        pair_data_device,
                        torch.stack(local_variants["candidate_matchings"], dim=0).float().to(self.device),
                        n1,
                        n2,
                    )

                    baseline_ged = int(baseline_ged_tensor[0].item())
                    candidate_geds_cpu = candidate_geds.detach().cpu().to(dtype=torch.int64)
                    best_local_ged = int(candidate_geds_cpu.min().item())
                    improvement = int(baseline_ged - best_local_ged)
                    processed_pairs += 1
                    shard_processed += 1
                    improvement_histogram[improvement] += 1
                    if improvement > 0:
                        improved_pairs += 1
                        shard_improved += 1
                    elif improvement == 0:
                        equal_pairs += 1

                    ged_histogram = defaultdict(int)
                    all_cases = []
                    for case_idx, ged_value in enumerate(candidate_geds_cpu.tolist()):
                        ged_histogram[int(ged_value)] += 1
                        if self.args.local_search_save_all_cases:
                            case_payload = dict(local_variants["candidate_records"][case_idx])
                            case_payload["ged"] = int(ged_value)
                            all_cases.append(case_payload)

                    topk_case_limit = max(int(self.args.local_search_topk_cases), 0)
                    sorted_case_indices = sorted(
                        range(len(local_variants["candidate_records"])),
                        key=lambda idx: (
                            int(candidate_geds_cpu[idx].item()),
                            local_variants["candidate_records"][idx]["assignment"],
                        ),
                    )
                    top_cases = []
                    for case_idx in sorted_case_indices[:topk_case_limit]:
                        case_payload = dict(local_variants["candidate_records"][case_idx])
                        case_payload["ged"] = int(candidate_geds_cpu[case_idx].item())
                        top_cases.append(case_payload)

                    pair_record = {
                        "pair_gid": [gid_1, gid_2],
                        "pred_ged_saved": float(artifact_item["pred_ged"]),
                        "gt_ged": float(artifact_item["gt_ged"]),
                        "baseline_ged_recomputed": baseline_ged,
                        "best_local_ged": best_local_ged,
                        "improvement": improvement,
                        "released_matches": local_variants["released_matches"],
                        "candidate_cols": [int(col_idx) for col_idx in local_variants["candidate_cols"]],
                        "extra_cols": [int(col_idx) for col_idx in local_variants["extra_cols"]],
                        "num_cases": int(len(local_variants["candidate_records"])),
                        "selected_probability_step": int(chosen_step_idx),
                        "selected_greedy_sample": int(chosen_sample_idx),
                        "ged_histogram": {str(key): int(value) for key, value in sorted(ged_histogram.items())},
                        "top_cases": top_cases,
                    }
                    if self.args.local_search_save_all_cases:
                        pair_record["all_cases"] = all_cases
                    pair_writer.write(json.dumps(pair_record) + "\n")

                per_shard_stats.append(
                    {
                        "artifact_path": str(artifact_path),
                        "pairs_in_shard": int(shard_total),
                        "processed_pairs": int(shard_processed),
                        "improved_pairs": int(shard_improved),
                        "duration_s": float(time.perf_counter() - shard_start),
                    }
                )
                if max_pairs > 0 and processed_pairs >= max_pairs:
                    break

        summary = {
            "artifact_paths": [str(path) for path in artifact_paths],
            "pairs_output_path": str(pairs_output_path),
            "processed_pairs": int(processed_pairs),
            "improved_pairs": int(improved_pairs),
            "equal_pairs": int(equal_pairs),
            "skipped_pairs": int(skipped_pairs),
            "improved_ratio": float(improved_pairs / processed_pairs) if processed_pairs > 0 else 0.0,
            "release_count": int(self.args.local_search_release_count),
            "extra_cols": int(self.args.local_search_extra_cols),
            "probability_step": int(self.args.local_search_probability_step),
            "save_all_cases": bool(self.args.local_search_save_all_cases),
            "topk_cases": int(self.args.local_search_topk_cases),
            "improvement_histogram": {str(key): int(value) for key, value in sorted(improvement_histogram.items())},
            "per_shard_stats": per_shard_stats,
        }
        with open(summary_output_path, "w") as summary_writer:
            json.dump(summary, summary_writer, indent=2)

        if self.is_main_process:
            print(f"[ArtifactLocalSearch] wrote pair records to {pairs_output_path}", flush=True)
            print(f"[ArtifactLocalSearch] wrote summary to {summary_output_path}", flush=True)





           
