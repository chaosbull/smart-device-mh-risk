# Author: ZengWenquan
# https://github.com/chaosbull
# License: CC BY 4.0

"""Settings for the run reported in the paper. Change these and the tables will move."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_social_media_mental_health.csv"
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DBIAN_CKPT = ROOT / "checkpoints" / "dbian_best.pt"

MODEL_NAME = "DRG-MDN-U"

EMBED_DIM = 8
EMBED_COLS = ["Country", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use"]
CAT_EMBED_DIM = len(EMBED_COLS) * EMBED_DIM

BASE_CONTINUOUS_COLS = [
    "Age", "Daily_Unlocks", "Study_Hours", "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night_bc", "Avg_Daily_Usage_Hours_bc", "Gender_num", "Stress_num",
]
DERIVED_COLS = [
    "Digital_Saturation",
    "Restorative_Ratio",
    "Behavioral_Balance",
    "Stress_Usage_Coupling",
]
CONTINUOUS_COLS = BASE_CONTINUOUS_COLS + DERIVED_COLS
CONTINUOUS_DIM = len(CONTINUOUS_COLS)
INPUT_DIM = CAT_EMBED_DIM + CONTINUOUS_DIM
TARGET_COL = "Mental_Health_Score"

STRESS_MAP = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Very High": 1.00}
GENDER_MAP = {"Female": 0, "Male": 1}
STRESS_FEATURE_IDX = CAT_EMBED_DIM + BASE_CONTINUOUS_COLS.index("Stress_num")

GRAPH_ATTN_DIM = 32
GRAPH_HEADS = 4
GCN_DIM = 32
GRAPH_TOPK = 6
GCN_LAYERS = 2
NODE_EMBED_DIM = 8
LOW_RANK_CROSS = 8
USE_FUSION = True

MDN_COMPONENTS = 3
MDN_HIDDEN = 64
MDN_DROPOUT = 0.06
POINT_HIDDEN = 64
REFINE_DIM = 40

# reported run: point head on, fusion gate off, calibrator applied only at eval
USE_POINT_HEAD = True
USE_ADAPTIVE_CALIBRATOR = True
USE_PREDICTIVE_FUSION = False
USE_MULTITASK_WEIGHTS = False
USE_REPRESENTATION_REFINE = False
USE_MULTISCALE_READOUT = True
USE_POINT_SKIP = True
CALIBRATOR_MIN_SCALE = 1.10  # sigma scale is at least this
CALIBRATE_AT_INFERENCE = True
SWA_START_EPOCH = 80
SWA_ENABLED = True

ENTROPY_GAMMA = 0.005
MSE_LOSS_WEIGHT = 0.58
MSE_POINT_WEIGHT = 0.06
MSE_AUX_WEIGHT = 0.0
CALIBRATION_WEIGHT = 0.0
CALIBRATION_TARGET = 0.90
LEARNABLE_SIGMA_SCALE = False
SIGMA_SCALE_INIT = 0.0

ADV_ETA = 0.03
ADV_EPS = 0.01
ADV_WARMUP_EPOCHS = 20

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 2e-4
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 40
EARLY_STOP_METRIC = "rmse"
GRAD_CLIP = 1.0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

GRAPH_DIM = GCN_DIM

from plot_style import (  # noqa: E402
    ACCENT, BG, CARD, CMAP_DIV, CMAP_SEQ, CMAP_UNCERT, C_PRIMARY, C_SECONDARY,
    C_TERTIARY, C_QUATERNARY, C_ACCENT, C_NEUTRAL, C_LIGHT, FONT_FAMILY,
    FONT_SIZE, GREEN, GRID, LABEL_SIZE, PALETTE, RED, SAVE_DPI, SPINE, TEXT,
    TICK_SIZE, TITLE_SIZE,
)
