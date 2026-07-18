# App-BMao t-limited

This directory is a parallel implementation of App-BMao with a real `-k/--stop` limit on expanded search states, adapted from the `DFS-BMao` t-limited implementation and applied to the A* BMao code path.

## What is different

- `./ged -k <t>` now stops A* or DFS after `t` state expansions.
- `-k <= 0` keeps the original full-search behavior.
- The `pair` mode still supports `-g` and `-x`, so the benchmark evaluation scripts can parse GED values and matchings as before.

## Build

```sh
make clean
make
```

This generates `./ged`.

## Usage

Show command-line help:

```sh
./ged -h
```

Run t-limited App-BMao on a graph pair file:

```sh
./ged -d graph_g.txt -q graph_q.txt -m pair -p astar -l BMao -g -x -k 100
```

## Benchmark evaluation

Single benchmark dataset:

```sh
python ./eval_ogbg_benchmark_pairs.py \
  --dataset ogbg-molhiv \
  --split test \
  --benchmark-root ../../artifacts/Overall_Performance/app_bmao_benchmark \
  --ged-bin ./ged \
  -k 100 \
  --batch-size 1 \
  --workers 1 \
  --timeout-seconds 30
```

All three benchmark datasets:

```sh
python ./eval_ogbg_benchmark_all.py \
  --split test \
  --benchmark-root ../../artifacts/Overall_Performance/app_bmao_benchmark \
  --ged-bin ./ged \
  -k 100 \
  --batch-size 1 \
  --workers 1 \
  --timeout-seconds 30
```
