import os
import sys
import pandas as pd
from pathlib import Path

# 현재 스크립트 위치 기준으로 프로젝트 루트(../../..) 찾아서 sys.path 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2] # src/f5_tts/scripts -> src/f5_tts -> src -> project_root
sys.path.insert(0, str(project_root))

from src.f5_tts.scripts.analyze_ngram_dist import analyze_dataset

def main():
    # Define target datasets
    # 폴더명이 정확해야 합니다. data/ 폴더를 기준으로 합니다.
    target_datasets = [
        "KSS_full_kor_grapheme",
        "KSS_full_kor_phoneme",
        "KSS_n2gk_allophone",
        "KSS_n2gk_i_only",
        "KSS_n2gk_c_only",
        "KSS_n2gk_n_only",
        "KSS_n2gk_i_and_c",
        "KSS_n2gk_i_and_n",
        "KSS_n2gk_inf",
        "KSS_n2gk_nf",
        "KSS_n2gk_efficient_allophone"
    ]

    data_root = Path(os.getcwd()) / "data"
    results = []

    print(f"Analyzing {len(target_datasets)} datasets...")
    
    for ds_name in target_datasets:
        if not (data_root / ds_name).exists():
            print(f"Skipping {ds_name} (Not found)")
            continue
            
        print(f"Processing: {ds_name}")
        try:
            stats = analyze_dataset(ds_name, data_root=data_root, silent=True)
            
            # Flatten results for table
            row = {"Dataset": ds_name.replace("KSS_", "").replace("n2gk_", "").replace("full_kor_", "")}
            
            # Add Unigram (1-gram) stats
            row["1-Vocab"] = stats[1]["vocab"]
            row["1-Gini"] = f"{stats[1]['gini']:.4f}"
            row["1-Renyi"] = f"{stats[1]['renyi']:.4f}"
            row["1-Eff"] = f"{stats[1]['eff']:.4f}"
            
            # Add Bigram (2-gram) stats
            row["2-Vocab"] = stats[2]["vocab"]
            row["2-Gini"] = f"{stats[2]['gini']:.4f}"
            row["2-Renyi"] = f"{stats[2]['renyi']:.4f}"
            row["2-Eff"] = f"{stats[2]['eff']:.4f}"
            
            # Add Trigram (3-gram) stats
            row["3-Vocab"] = stats[3]["vocab"]
            row["3-Gini"] = f"{stats[3]['gini']:.4f}"
            row["3-Renyi"] = f"{stats[3]['renyi']:.4f}"
            row["3-Eff"] = f"{stats[3]['eff']:.4f}"
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")

    # Create DataFrame and Print Markdown
    if results:
        df = pd.DataFrame(results)
        
        # Column order
        cols = ["Dataset", 
                "1-Vocab", "1-Gini", "1-Renyi", "1-Eff",
                "2-Vocab", "2-Gini", "2-Renyi", "2-Eff",
                "3-Vocab", "3-Gini", "3-Renyi", "3-Eff"]
        
        print("\n" + "="*50)
        print("Analysis Summary (1/2/3-gram) [Renyi alpha=2.5]")
        print("="*50)
        # print(df.to_markdown(index=False)) # tabulate dependency missing
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = "analysis_ngram_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
