import pandas as pd

# Extract spleen annotations from annotations_facs.csv
def extract_spleen_rows(in_path, out_path, tissue="Spleen"):
    df = pd.read_csv(in_path, low_memory=False)
    selected_rows = df[df["tissue"] == tissue]
    selected_rows.to_csv(out_path, index=False)
    return selected_rows

extract_spleen_rows("data/annotations_facs.csv","data/annotations_facs_spleen.csv","Spleen")