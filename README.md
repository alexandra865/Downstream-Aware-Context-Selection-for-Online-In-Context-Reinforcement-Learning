# Downstream-Aware Context Selection for Online In-Context Reinforcement Learning

Core implementation for our [paper](https://arxiv.org/abs/2609.33166).

## Overview

In-context reinforcement learning (ICRL) lets LLM agents adapt through past
interactions without updating their parameters. As histories grow, repeatedly
including them in the prompt increases token cost. Our framework manages a
bounded history using the full rolling window as its reference. A predictor
estimates the downstream effect of removing historical interactions, determines
a deletion order, and selects a context budget for each decision.

## Code Components

- **Context selection (`code/`)**: History alignment, embeddings, structured
  features, predictor inference, Nested-Borda, and adaptive budget selection.
- **Learned order-guided redundancy compression (LORC)** (`h20_ours_runtime.py`,
  `h20_ours_renderer.py`, `h20_scene_parser.py`): Use the deletion order to choose
  where repeated static field values are retained, while preserving dynamic
  information, actions, and rewards.
- **Method entrypoints (`ICRLCloseLoop_h20_*.py`)**: Closed-loop driving with
  Ours, Full-H20, Recent-K, or Similarity-K.
- **Simulation (`simInfo/`, `simModel/`, `formal_domain_config.py`)**:
  Observations, prompts, scene configuration, SUMO interaction, and replay.
- **Planning and evaluation (`trafficManager/`, `DriverAgent/Evaluation.py`)**:
  Vehicle prediction, trajectory planning, and Safety, Comfort, and Efficiency
  scoring.

## Installation

- Python 3.10
- SUMO >= 1.15.0
- CARLA 0.9.15 Python API (included in `requirements.txt`)
- A desktop session for the simulation viewer

Create the environment and install the dependencies:

```bash
conda create -n icrl-context python=3.10
conda activate icrl-context
python -m pip install -r requirements.txt pandas chromadb
```

Set `SUMO_HOME` to your SUMO installation:

```bash
export SUMO_HOME="/path/to/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
sumo --version
```

## API Configuration

Set the following variables in the terminal used to run the project:

```bash
export OPENAI_API_TYPE="openai"
export OPENAI_API_KEY="your-openai-api-key"
```

The default driving policy is `gpt-4o`. The adaptive selector uses
`text-embedding-3-large` for history and query embeddings.

## Running

Run these commands from the project root. The included scene uses `Town05`,
route `town05_default`, and ego vehicle `50`. Create an output directory:

```bash
RUN_DIR="results/town05_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
```

### Ours

Set `--final-adaptive-selector-dir` to your selector directory containing
`h20_selector_artifact.json`. The configuration specifies the ranking artifact
and five predictor checkpoints. You can also pass the artifact JSON path directly:

```bash
python ICRLCloseLoop_h20_ours.py \
  --context-policy adaptive_h20 \
  --final-adaptive-selector-dir /path/to/selector_artifact \
  --pair-id town05_clear1x_seed0 \
  --domain-id clear_1x --sumo-seed 0 \
  --output-prefix "$RUN_DIR/ours"
```

### Full-H20

```bash
python ICRLCloseLoop_h20_full_context_diagnostic.py \
  --context-policy full_h20 --pair-id town05_clear1x_seed0 \
  --domain-id clear_1x --sumo-seed 0 \
  --output-prefix "$RUN_DIR/full_h20"
```

### Recent-K

```bash
python ICRLCloseLoop_h20_recent_k_strict_h20.py \
  --context-policy recent_h20 --recent-k 13 \
  --pair-id town05_clear1x_seed0 \
  --domain-id clear_1x --sumo-seed 0 \
  --output-prefix "$RUN_DIR/recent_13"
```

### Similarity-K

Place your feature-normalization parameters at
`results/h20_statesim_baseline_v1/statesim_feature_scaler.json`:

```bash
python ICRLCloseLoop_h20_statesim.py \
  --context-policy statesim_h20 --recent-k 13 \
  --pair-id town05_clear1x_seed0 \
  --domain-id clear_1x --sumo-seed 0 \
  --output-prefix "$RUN_DIR/similarity_13"
```

Similarity-K uses structured-state nearest-neighbor retrieval and supports
K = 13, 14, and 15. For method comparisons, use the same `--pair-id`, scene,
domain, seed, and policy settings.

Add `--sumo-gui` to open SUMO-GUI, or `--max-decisions 3` for a short run.
Use an entrypoint's `--help` for the full option list.

## Outputs

Each run saves a simulation database (`.db`), episode metrics
(`_per_episode_metrics.csv`), selection logs (`_per_decision_selector_log.csv`),
context records (`_paired_context_log.jsonl`), episode status
(`_episode_status.json`), and scene configuration (`_domain_manifest.json`).
Metrics include Safety, Comfort, Efficiency, selected K, and policy token usage.

## Acknowledgments

The driving framework builds on
[ICRL_Autonomous_Driving](https://github.com/rohanchandra30/ICRL_Autonomous_Driving)
and [LimSim_plus](https://github.com/PJLab-ADG/LimSim/tree/LimSim_plus).

## License

GNU General Public License, version 3. See `LICENSE`.
