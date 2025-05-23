import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
xlsx_dir = BASE_DIR / "data" / "xlsx" / "final_review.xlsx"
csv_dir = BASE_DIR / "data" / "csv" / "final_review_processed.csv"
df = pd.read_excel(xlsx_dir)
headers = df.iloc[0]
new_df  = pd.DataFrame(df.values[1:], columns=headers)
new_df.to_csv(csv_dir,sep=";")

