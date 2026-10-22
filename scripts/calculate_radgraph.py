import json
import torch
import numpy as np
from radgraph import F1RadGraph
import argparse
import sys
from tqdm import tqdm

def calculate_radgraph_f1(generated_path, reference_path, model_type="radgraph", reward_level="all", batch_size=8):
    print(f"Loading generated reports from: {generated_path}")
    print(f"Loading reference reports from: {reference_path}")
    
    with open(generated_path, 'r', encoding='utf-8') as f:
        gen_lines = f.readlines()
        
    with open(reference_path, 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()
        
    if len(gen_lines) != len(ref_lines):
        print(f"Warning: Number of lines mismatch (Gen: {len(gen_lines)}, Ref: {len(ref_lines)})")
        min_len = min(len(gen_lines), len(ref_lines))
        gen_lines = gen_lines[:min_len]
        ref_lines = ref_lines[:min_len]
    
    # Pre-clean the reports
    gen_reports = []
    ref_reports = []
    
    for gen, ref in zip(gen_lines, ref_lines):
        gen_clean = gen.strip().strip("[]").strip("'").strip('"')
        
        # Handle reference parsing (taking first report if multiple)
        try:
            r_list = eval(ref.strip())
            if isinstance(r_list, list) and len(r_list) > 0:
                ref_clean = r_list[0].strip()
            else:
                ref_clean = str(r_list).strip()
        except:
            ref_clean = ref.strip().strip("[]").strip("'").strip('"')
            
        gen_reports.append(gen_clean)
        ref_reports.append(ref_clean)
        
    print(f"Initializing F1RadGraph with model: {model_type}...")
    f1_scorer = F1RadGraph(reward_level=reward_level, model_type=model_type)
    
    print(f"Calculating RadGraph F1 for {len(gen_reports)} samples...")
    # The F1RadGraph forward pass can take lists. 
    # To avoid memory issues or time-outs, we process in chunks.
    
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    # Process in chunks of 50 for safety
    chunk_size = 50
    for i in tqdm(range(0, len(gen_reports), chunk_size)):
        chunk_gen = gen_reports[i:i+chunk_size]
        chunk_ref = ref_reports[i:i+chunk_size]
        
        try:
            mean_reward, reward_list, _, _ = f1_scorer(hyps=chunk_gen, refs=chunk_ref)
            
            if reward_level == "all":
                # mean_reward is (P, R, F1)
                all_precisions.append(mean_reward[0])
                all_recalls.append(mean_reward[1])
                all_f1s.append(mean_reward[2])
            else:
                all_f1s.append(mean_reward)
        except Exception as e:
            print(f"Error in chunk {i}: {e}")
            continue
            
    final_p = np.mean(all_precisions) if all_precisions else 0
    final_r = np.mean(all_recalls) if all_recalls else 0
    final_f1 = np.mean(all_f1s) if all_f1s else 0
    
    print("\n" + "="*30)
    print("RadGraph Evaluation Result")
    print("="*30)
    print(f"Precision: {final_p:.4f}")
    print(f"Recall:    {final_r:.4f}")
    print(f"F1 Score:  {final_f1:.4f}")
    print("="*30)
    
    return final_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', type=str, default='outputs/generated_reports (1).txt')
    parser.add_argument('--ref', type=str, default='outputs/reference_reports (1).txt')
    parser.add_argument('--model_type', type=str, default='radgraph')
    parser.add_argument('--chunk_size', type=int, default=50)
    args = parser.parse_args()
    
    calculate_radgraph_f1(args.gen, args.ref, model_type=args.model_type)
