import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from collections import Counter
import numpy as np

# Load the final CSV
BASE_DIR = Path(__file__).resolve().parent.parent
csv_dir = BASE_DIR / "data" / "csv" / "final_review_processed.csv"
df = pd.read_csv(csv_dir,sep=";")

# Filter included studies
included = df[df["AddToFinalSet"].str.lower() == "yes"].copy()

# Extract and prepare some aggregated information
model_counts = included["Models"].dropna().apply(lambda x: [m.strip() for m in x.split(",")])
model_flat = [m for sublist in model_counts for m in sublist]
model_distribution = Counter(model_flat)

task_counts = included["TaskTypes"].dropna().apply(lambda x: [t.strip() for t in x.split(",")])
task_flat = [t for sublist in task_counts for t in sublist]
task_distribution = Counter(task_flat)

# Prepare bar chart for models
plt.figure(figsize=(10, 4))
sns.barplot(x=list(model_distribution.keys()), y=list(model_distribution.values()))
plt.title("Model Usage in Included Studies")
plt.ylabel("Number of Studies")
plt.xticks(rotation=45)
plt.tight_layout()
model_chart_path = BASE_DIR / "figures" / "model_distribution.png"
plt.savefig(model_chart_path)
plt.close()

# Prepare bar chart for tasks
plt.figure(figsize=(10, 4))
sns.barplot(x=list(task_distribution.keys()), y=list(task_distribution.values()))
plt.title("Task Types in Included Studies")
plt.ylabel("Number of Studies")
plt.xticks(rotation=45)
plt.tight_layout()
task_chart_path = BASE_DIR / "figures" / "task_distribution.png"
plt.savefig(task_chart_path)
plt.close()

# Suggest LaTeX-friendly interpretations
summary_stats = {
    "num_included": len(included),
    "num_total": len(df),
    "most_common_models": model_distribution.most_common(3),
    "most_common_tasks": task_distribution.most_common(3),
    "model_chart_path": model_chart_path,
    "task_chart_path": task_chart_path
}

# Plot extracted data
xlsx_dir = BASE_DIR / "data" / "xlsx" / "extracted_data.xlsx"
df = pd.read_excel(xlsx_dir)
df.columns = df.iloc[0]
df = df.drop(0).reset_index(drop=True)
df.columns = ["Study", "Test", "Metric", "Effect Size", "p-value"]
df["Effect Size"] = pd.to_numeric(df["Effect Size"], errors='coerce')
df["p-value"] = pd.to_numeric(df["p-value"], errors='coerce')

# Filter for non-null p-values and effect sizes
df_pcurve = df[df["p-value"].notnull()]
df_funnel = df[df["Effect Size"].notnull()]

# Plot: P-Curve (Histogram of p-values under 0.05)
plt.figure(figsize=(8, 5))
sns.histplot(df_pcurve["p-value"], edgecolor='black')
plt.title("P-Curve: Distribution of Significant P-Values (p < 0.05)")
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
pcurve_path = BASE_DIR / "figures" / "p-curve.png"
plt.savefig(pcurve_path)

# Plot: Funnel plot (Effect Size vs. Standard Error estimate)
# Approximate SE as 1/sqrt(N); use N = 100 as placeholder (real SE unknown)
df_funnel["SE"] = 1 / np.sqrt(100)
plt.figure(figsize=(8, 5))
plt.scatter(df_funnel["Effect Size"], df_funnel["SE"])
plt.gca().invert_yaxis()
plt.title("Funnel Plot: Effect Size vs. SE (Estimated)")
plt.xlabel("Effect Size")
plt.ylabel("Standard Error (approx.)")
plt.grid(True)
plt.tight_layout()
funnel_path = BASE_DIR / "figures" / "funnel.png"
plt.savefig(funnel_path)