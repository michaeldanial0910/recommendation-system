"""Paths and run settings shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ml-1m"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Official GroupLens file (https://files.grouplens.org/datasets/movielens/ml-1m.zip).
ZIP_MD5 = "c4d9eecfca2ab87c1945afe126590906"

SEED = 42
# Global time cutoffs as quantiles of all rating timestamps. Everything before the first
# cutoff is train, between the two is validation (used only for tuning), after is test.
# A global cutoff (not "last N per user") means no rating from the future, by anyone,
# is ever visible to the model: the same situation as a model deployed on that date.
VAL_QUANTILE = 0.70
TEST_QUANTILE = 0.80

K = 10                 # length of a recommendation list
RELEVANT_MIN = 4       # a test rating of 4 or 5 counts as a hit
ONBOARDING_PICKS = 3   # liked movies a new user "picks" in the cold-start onboarding scenario
WARM_MIN_HISTORY = 5   # users with fewer train ratings are routed to the cold-start path

AGE_GROUPS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student",
               "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
               "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
               "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]
