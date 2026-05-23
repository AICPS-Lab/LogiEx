import os
import sys
import json
import time
import random

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, 'hscc-iccps26-paper254-repeatability-evaluations')
sys.path.insert(0, _BACKEND_ROOT)
from backend.tasks.transit_logics import parse
from backend.tasks.logic_parameterizer import quantitativescore, qualitativescore, count_nodes
from eval_query_classify import get_logic_gt, BASE_LEVEL, SECOND_LEVEL, LOGIC_COMPARISON


SECOND_LEVEL_FOR_RUNTIME = [t for t in SECOND_LEVEL if t not in (18, 22, 23, 24)]

TABLE6_CATEGORIES = {
    "Base-Level": BASE_LEVEL,
    "Second-Level": SECOND_LEVEL_FOR_RUNTIME,
    "Logic Comparison": LOGIC_COMPARISON,
}

TREE_SIZES = [23, 42, 50, 1086, 1500]
K_VALUES = [10, 50]


def scan_all_trees(data_dirs):
    all_trees = {}
    for data_dir in data_dirs:
        data_path = os.path.expanduser(data_dir)
        if not os.path.exists(data_path):
            continue
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        size = count_nodes(data)
                        if size not in all_trees:
                            all_trees[size] = []
                        all_trees[size].append(filepath)
                except Exception as e:
                    continue
    return all_trees


def find_trees_for_target_sizes(all_trees, target_sizes):   
    matched = {}
    for target_size in target_sizes:
        if target_size in all_trees:
            matched[target_size] = all_trees[target_size][0]
        else:
            closest_size = min(all_trees.keys(), key=lambda x: abs(x - target_size))
            matched[target_size] = all_trees[closest_size][0]
            print(f"  Target {target_size}: using closest match size {closest_size} from {os.path.basename(matched[target_size])}")
    return matched


def generate_logic_queries_for_category(category_types, k, scenario_num=0):
    queries = []
    for i in range(k):
        query_type = category_types[i % len(category_types)]
        if query_type in [1, 2, 6, 15, 22, 30, 31]:
            value = [-1, -1]
        elif query_type in [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 23, 24, 25, 26, 27, 29]:
            value = [random.randint(0, 4), -1]
        else:
            value = [random.randint(0, 4), random.randint(0, 4)]
        logic_str = get_logic_gt(query_type, value)
        queries.append(logic_str)
    return queries


def evaluate_logic_queries(queries, tree_data, scenario_num=0):
    start_time = time.time()
    for logic_str in queries:
        logic_parts = logic_str.split("; ")
        for logic_part in logic_parts:
            logic_part = logic_part.strip()
            try:
                parsed = parse(logic_part)
                try:
                    quantitativescore(parsed, tree_data, scenario_num)
                except TypeError:
                    pass
                try:
                    qualitativescore(parsed, tree_data, scenario_num)
                except TypeError:
                    pass
            except Exception:
                pass
    elapsed = time.time() - start_time
    return elapsed


def run_table6_eval():
    data_dirs = [
        os.path.join(_BACKEND_ROOT, 'backend', 'data', 'transit_eval_json'),
        os.path.join(_BACKEND_ROOT, 'backend', 'data', 'transit_additional_search'),
    ]
    
    print("Scanning existing trees in data directories...")
    all_trees = scan_all_trees(data_dirs)
    
    tree_mapping = find_trees_for_target_sizes(all_trees, TREE_SIZES)
    
    results = {}
    random.seed(42)
    
    for target_size in TREE_SIZES:
        tree_file = tree_mapping[target_size]
        with open(tree_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        actual_size = count_nodes(tree_data)
        print(f"\nProcessing tree (size={actual_size}) from {os.path.basename(tree_file)}")
        
        results[target_size] = {}
        for k in K_VALUES:
            results[target_size][k] = {}
            for cat_name, category_types in TABLE6_CATEGORIES.items():
                queries = generate_logic_queries_for_category(category_types, k)
                runtime = evaluate_logic_queries(queries, tree_data)
                results[target_size][k][cat_name] = runtime
                print(f"  {cat_name} k={k}: {runtime:.4f}s")
    
    print("\n" + "-" * 100)
    header = f"{'Tree Size':<12} | {'Base-Level':<25} | {'Second-Level':<25} | {'Logic Comparison':<25}"
    print(header)
    subheader = f"{'':<12} | {'k=10':<12} {'k=50':<12} | {'k=10':<12} {'k=50':<12} | {'k=10':<12} {'k=50':<12}"
    print(subheader)
    print("-" * 100)
    for tree_size in TREE_SIZES:
        if tree_size not in results:
            continue
        row = f"{tree_size:<12} |"
        for cat_name in ["Base-Level", "Second-Level", "Logic Comparison"]:
            k10 = results[tree_size][10][cat_name] if 10 in results[tree_size] else 0.0
            k50 = results[tree_size][50][cat_name] if 50 in results[tree_size] else 0.0
            row += f" {k10:.4f}  {k50:.4f} |"
        print(row)
    print("-" * 100)


if __name__ == "__main__":
    run_table6_eval()
