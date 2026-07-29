"""Global configuration for the heterogeneous credit assignment baseline."""
import os
from pathlib import Path
import torch

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
for d in (DATA, RESULTS):
    d.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MR_DEVICE overrides autodetect (escape hatch when MPS is wedged: MR_DEVICE=cpu).
DEVICE = os.getenv("MR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16
ATTN_IMPL = "eager"

SEED = 0

# Evaluation batch sizes. Sized against measured MPS throughput (~4.3 ex/s at
# seqlen 68) so that one full evaluation costs roughly 15 seconds.
TASK_EVAL_N = 40
RETENTION_EVAL_N = 24
FAILURE_POOL_N = 32
FAILURE_BATCH_N = 12
FORWARD_BATCH = 16

# Experiment scale
N_COLLECT_STATES = 30
N_ROUTING_EPISODES = 4
N_ROUTING_ROUNDS = 3
CANDIDATES_PER_TYPE = 3

# Cost model: normalized units. Reflects wall-clock expense of realizing a
# candidate (apply + evaluate), which is what a budgeted selector must respect.
COST_PROMPT = 1.0
COST_ACTIVATION = 2.0
COST_LORA = 8.0
ROUTING_BUDGET = 12.0  # per routing round

# Utility weighting: retention loss is penalized, so a candidate that wins on
# task accuracy but damages general capability is not automatically preferred.
RETENTION_PENALTY = 1.0
