from src.plotting import plot_inference_results_all, plot_inference_results_individual
from src.analysis import summarize_inference_results
import os
import pandas as pd

INFERENCE_DIR = "./outputs/co_co2_cv_20260122_1607/Inference/"

inf_df = pd.read_csv(
    os.path.join(INFERENCE_DIR, "inference_results.csv"), delimiter=","
)

plot_inference_results_all(inf_df, INFERENCE_DIR)
plot_inference_results_individual(inf_df, INFERENCE_DIR)
summarize_inference_results(inf_df, INFERENCE_DIR)
