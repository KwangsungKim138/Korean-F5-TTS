import os
import re
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
DATA_ROOT = "data/KSS"
TEST_TXT_PATH = os.path.join(DATA_ROOT, "test.txt")
TRAIN_TXT_PATH = os.path.join(DATA_ROOT, "train_full.txt")
OUTPUT_FILE = "reference_mapping_result.txt"

# Constraints (Same as evaluate_models.py)
MIN_DURATION = 2.7
CHAR_DURATION_RATIO = 0.33

# Fallback check for root files
if not os.path.exists(TEST_TXT_PATH) and os.path.exists("test.txt"):
    TEST_TXT_PATH = "test.txt"
if not os.path.exists(TRAIN_TXT_PATH) and os.path.exists("train_full.txt"):
    TRAIN_TXT_PATH = "train_full.txt"

def parse_kss_line(line):
    parts = line.strip().split("|")
    if len(parts) < 5:
        return None
    try:
        return {
            "path": parts[0],
            "text": parts[2], # Expanded text
            "duration": float(parts[4])
        }
    except ValueError:
        return None

def get_pure_char_count(text):
    # Count meaningful characters (Hangul, Alphabet, Number)
    return len(re.findall(r'[가-힣A-Za-z0-9]', text))

def get_target_punctuation(text):
    text = text.strip()
    if not text: return "."
    last_char = text[-1]
    if last_char in ['.', '?', '!']:
        return last_char
    return "."

def is_valid_candidate(text):
    # 1. No commas
    if ',' in text:
        return False
    # 2. Period allowed ONLY at the very end
    if '.' in text[:-1]:
        return False
    return True

def main():
    print(f"Reading Test Set: {TEST_TXT_PATH}")
    print(f"Reading Train Set: {TRAIN_TXT_PATH}")

    if not os.path.exists(TEST_TXT_PATH) or not os.path.exists(TRAIN_TXT_PATH):
        print("Error: Dataset files not found.")
        return

    # 1. Load Data
    test_items = []
    with open(TEST_TXT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = parse_kss_line(line)
            if item:
                test_items.append(item)

    candidates = []
    with open(TRAIN_TXT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = parse_kss_line(line)
            if item and is_valid_candidate(item['text']):
                candidates.append(item)
    
    print(f"\n[Stats]")
    print(f"  - Test items loaded: {len(test_items)}")
    print(f"  - Valid Candidates: {len(candidates)}")
    print("-" * 40)
    print(f"Target Duration Logic (Synced with evaluate_models.py):")
    print(f"  Target = max({MIN_DURATION}, CharCount * {CHAR_DURATION_RATIO})")
    print(f"Matching Priority:")
    print(f"  1. Strict (Chars + Punct) -> Closest Duration")
    print(f"  2. Char Match -> Closest Duration")
    print(f"  3. Fallback -> Closest Duration")
    print("-" * 40)

    results = []
    match_stats = {}
    
    for t_item in tqdm(test_items):
        t_chars = get_pure_char_count(t_item['text'])
        t_punct = get_target_punctuation(t_item['text'])
        
        # Calculate Ideal Target Duration
        calc_dur = t_chars * CHAR_DURATION_RATIO
        target_dur = max(calc_dur, MIN_DURATION)
        
        # Pool Selection
        match_type = "Fallback"
        
        # 1. Try Strict Match (Chars + Punct)
        strict_pool = [c for c in candidates 
                       if get_pure_char_count(c['text']) == t_chars 
                       and get_target_punctuation(c['text']) == t_punct]
        
        if strict_pool:
            pool = strict_pool
            match_type = "Strict"
        else:
            # 2. Try Char Match only
            char_pool = [c for c in candidates if get_pure_char_count(c['text']) == t_chars]
            if char_pool:
                pool = char_pool
                match_type = "CharMatch"
            else:
                # 3. Fallback to all
                pool = candidates
                match_type = "Fallback"
        
        # Find best duration match in pool
        best_cand = min(pool, key=lambda c: abs(c['duration'] - target_dur))
        
        match_stats[match_type] = match_stats.get(match_type, 0) + 1
        
        results.append({
            "test": t_item,
            "ref": best_cand,
            "match_type": match_type,
            "target_dur": target_dur,
            "t_chars": t_chars
        })

    # 3. Save to TXT
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Reference Mapping Result (Synced with evaluate_models.py)\n")
        f.write(f"Total Test Items: {len(results)}\n")
        f.write(f"Stats: {match_stats}\n")
        f.write("="*60 + "\n\n")
        
        for i, res in enumerate(results):
            t = res['test']
            r = res['ref']
            m_type = res['match_type']
            tgt = res['target_dur']
            t_chars = res['t_chars']
            
            f.write(f"[Test #{i+1}] {t['path']}\n")
            f.write(f"  Text: \"{t['text']}\" (Chars: {t_chars}, Punct: '{get_target_punctuation(t['text'])}')\n")
            f.write(f"  Orig Dur: {t['duration']}s -> Target Dur: {tgt:.2f}s (Min {MIN_DURATION}s)\n")
            
            if r:
                r_chars = get_pure_char_count(r['text'])
                r_punct = get_target_punctuation(r['text'])
                diff = r['duration'] - tgt
                
                f.write(f"  => Matched Ref [{m_type}]: {r['path']} (Dur: {r['duration']}s)\n")
                f.write(f"     Ref Text: \"{r['text']}\" (Chars: {r_chars}, Punct: '{r_punct}')\n")
                f.write(f"     Diff from Target: {diff:+.2f}s\n")
            else:
                f.write(f"  => No suitable reference found!\n")
            
            f.write("-" * 50 + "\n")
            
    print(f"Done! Please check '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
