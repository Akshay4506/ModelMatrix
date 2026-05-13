import os
import pandas as pd
from pathlib import Path

def merge_datasets():
    # Paths
    base_dir = Path(__file__).parent.parent
    src_dir = base_dir / "datasets"
    out_dir = base_dir / "datasets_merged"
    
    # Create output directory
    out_dir.mkdir(exist_ok=True)
    
    print(f"Searching for datasets in {src_dir}...")
    
    # Find all _X.csv files
    x_files = list(src_dir.glob("*_X.csv"))
    
    if not x_files:
        print("No _X.csv files found!")
        return

    for x_path in x_files:
        dataset_name = x_path.name.replace("_X.csv", "")
        y_path = src_dir / f"{dataset_name}_y.csv"
        
        if y_path.exists():
            print(f"Merging {dataset_name}...")
            
            # Load X and y
            df_x = pd.read_csv(x_path)
            df_y = pd.read_csv(y_path)
            
            # Combine
            # We assume y has one column. If it has a header, we keep it, otherwise name it 'target'
            target_col = df_y.columns[0] if len(df_y.columns) > 0 else "target"
            df_x[target_col] = df_y.iloc[:, 0]
            
            # Save
            out_path = out_dir / f"{dataset_name}.csv"
            df_x.to_csv(out_path, index=False)
            print(f"  -> Saved to {out_path}")
        else:
            print(f"Skipping {dataset_name} (y file missing)")

    print("\nMerge complete! You can now find the combined files in the 'datasets_merged' folder.")

if __name__ == "__main__":
    merge_datasets()
