import json
from pycocoevalcap.cider.cider import Cider
import argparse

def calculate_cider(generated_path, reference_path):
    print(f"Loading generated reports from: {generated_path}")
    print(f"Loading reference reports from: {reference_path}")
    
    with open(generated_path, 'r', encoding='utf-8') as f:
        gen_lines = f.readlines()
        
    with open(reference_path, 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()
        
    if len(gen_lines) != len(ref_lines):
        print(f"Warning: Number of lines mismatch (Gen: {len(gen_lines)}, Ref: {len(ref_lines)})")
        # Truncate to min length
        min_len = min(len(gen_lines), len(ref_lines))
        gen_lines = gen_lines[:min_len]
        ref_lines = ref_lines[:min_len]
    
    # Format for CIDEr (dict mapping image_id to list of sentences)
    res = {}
    gts = {}
    
    for i, (gen, ref) in enumerate(zip(gen_lines, ref_lines)):
        gen_clean = gen.strip().strip("[]").strip("'").strip('"')
        ref_clean = ref.strip().strip("[]").strip("'").strip('"')
        
        try:
            r_list = eval(ref.strip())
            if isinstance(r_list, list):
                ref_clean_list = [r.strip() for r in r_list]
            else:
                ref_clean_list = [str(r_list).strip()]
        except:
            ref_clean_list = [ref_clean]
            
        res[i] = [gen_clean]
        gts[i] = ref_clean_list
        
    print(f"Processing {len(res)} samples...")
    
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    
    print("\n" + "="*30)
    print(f"CIDEr Score: {score:.4f}")
    print("="*30)
    
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', type=str, default='outputs/generated_reports (1).txt')
    parser.add_argument('--ref', type=str, default='outputs/reference_reports (1).txt')
    args = parser.parse_args()
    
    calculate_cider(args.gen, args.ref)
