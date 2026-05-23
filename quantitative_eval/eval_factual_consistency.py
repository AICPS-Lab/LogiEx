from transformers import BertForSequenceClassification, BertTokenizer
from bert_score import score
import torch.nn as nn
from collections import defaultdict
import numpy as np
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, 'hscc-iccps26-paper254-repeatability-evaluations')
sys.path.insert(0, _BACKEND_ROOT)

BASE_LEVEL = [1, 2, 3, 5, 6, 15, 30, 31]
SECOND_LEVEL = [4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 25, 26, 27, 29]
LOGIC_COMPARISON = [19, 20, 21, 28]
TABLE3_CATEGORIES = {
    "Base-Level": BASE_LEVEL,
    "Second-Level": SECOND_LEVEL,
    "Logic Comparison": LOGIC_COMPARISON,
}




# def bert_score_eval():
#     model_path = 'bert-large-uncased'
#     file_path = os.path.join(_EVAL_DIR, 'eval_result')
#     # gpt_output_file = f"{file_path}/factual_gpt_result.txt"
#     gpt_output_file = f"{file_path}/vanilla_llm_result.txt"
#     narrative_gt_file = f"{file_path}/factual_narrative_gt.txt"
#     question_type_file = f"{file_path}/factual_gt.txt"
    
#     with open(gpt_output_file, 'r') as gpt_file, open(narrative_gt_file, 'r') as narrative_file, open(question_type_file, 'r') as query_type_file:
#         text_list = gpt_file.readlines()
#         summary_list = narrative_file.readlines()
#         query_types = [int(qt.strip()) for qt in query_type_file.readlines()]

#     if len(text_list) != len(summary_list) or len(text_list) != len(query_types):
#         raise ValueError("Mismatch between the number of lines in gpt_output_file, narrative_gt, and question types.")

#     half_len = len(text_list) // 2
#     run1_texts = text_list[:half_len]
#     run2_texts = text_list[half_len:]
#     run1_summaries = summary_list[:half_len]
#     run2_summaries = summary_list[half_len:]

#     query_types_run1 = query_types[:half_len]
#     query_types_run2 = query_types[half_len:]

#     run1_results = defaultdict(list)
#     run2_results = defaultdict(list)
#     combined_results = defaultdict(list)

#     def calculate_bertscore_for_runs(srcs_run1, tgts_run1, srcs_run2, tgts_run2):
#         P1, R1, F1_run1 = score(srcs_run1, tgts_run1, model_type=model_path, verbose=True)
#         P2, R2, F1_run2 = score(srcs_run2, tgts_run2, model_type=model_path, verbose=True)

#         F1_run1 = F1_run1.tolist()
#         F1_run2 = F1_run2.tolist()
        
#         combined_F1 = [max(f1_r1, f1_r2) for f1_r1, f1_r2 in zip(F1_run1, F1_run2)]
        
#         return F1_run1, F1_run2, combined_F1

#     for category in set(query_types):
#         run1_texts_cat = [text for text, qt in zip(run1_texts, query_types_run1) if qt == category]
#         run1_summaries_cat = [summary for summary, qt in zip(run1_summaries, query_types_run1) if qt == category]

#         run2_texts_cat = [text for text, qt in zip(run2_texts, query_types_run2) if qt == category]
#         run2_summaries_cat = [summary for summary, qt in zip(run2_summaries, query_types_run2) if qt == category]

#         F1_run1, F1_run2, combined_F1 = calculate_bertscore_for_runs(run1_texts_cat, run1_summaries_cat, run2_texts_cat, run2_summaries_cat)

#         run1_results[category] = F1_run1
#         run2_results[category] = F1_run2
#         combined_results[category] = combined_F1

#     overall_F1_run1 = sum(run1_results.values(), [])
#     overall_F1_run2 = sum(run2_results.values(), [])
#     overall_combined_F1 = sum(combined_results.values(), [])

#     def calculate_accuracy(f1_scores, threshold=0.57):
#         correct_predictions = sum(1 for f1 in f1_scores if f1 >= threshold)
#         accuracy = (correct_predictions / len(f1_scores)) * 100
#         return accuracy

#     run1_accuracy = calculate_accuracy(overall_F1_run1)
#     run2_accuracy = calculate_accuracy(overall_F1_run2)
#     combined_accuracy = calculate_accuracy(overall_combined_F1)

#     print(f"Accuracy for Run 1 (Threshold 0.7): {run1_accuracy:.2f}%")
#     print(f"Accuracy for Run 2 (Threshold 0.7): {run2_accuracy:.2f}%")
#     print(f"Combined Accuracy (Threshold 0.7): {combined_accuracy:.2f}%")

#     for category in set(query_types):
#         avg_F1_cat_run1 = sum(run1_results[category]) / len(run1_results[category])
#         avg_F1_cat_run2 = sum(run2_results[category]) / len(run2_results[category])
#         avg_combined_F1_cat = sum(combined_results[category]) / len(combined_results[category])

#         run1_cat_accuracy = calculate_accuracy(run1_results[category])
#         run2_cat_accuracy = calculate_accuracy(run2_results[category])
#         combined_cat_accuracy = calculate_accuracy(combined_results[category])

#         print(f"Category {category} - Run 1 Accuracy: {run1_cat_accuracy:.2f}%")
#         print(f"Category {category} - Run 2 Accuracy: {run2_cat_accuracy:.2f}%")
#         print(f"Category {category} - Combined Accuracy: {combined_cat_accuracy:.2f}%")


from collections import defaultdict
from bert_score import score

def bert_score_eval():
    model_path = 'bert-large-uncased'
    file_path = os.path.join(_EVAL_DIR, 'eval_result')
    gpt_output_file = f"{file_path}/factual_llama_baseline_result.txt"
    narrative_gt_file = f"{file_path}/factual_narrative_gt.txt"
    question_type_file = f"{file_path}/factual_gt.txt"
    
    with open(gpt_output_file, 'r') as gpt_file, open(narrative_gt_file, 'r') as narrative_file, open(question_type_file, 'r') as query_type_file:
        text_list = gpt_file.readlines()
        summary_list = narrative_file.readlines()
        query_types = [int(qt.strip()) for qt in query_type_file.readlines()]

    if len(text_list) != len(summary_list) or len(text_list) != len(query_types):
        raise ValueError("Mismatch between the number of lines in gpt_output_file, narrative_gt, and question types.")

    # Split the data into 3 parts
    third_len = len(text_list) // 3
    run1_texts = text_list[:third_len]
    run2_texts = text_list[third_len:2*third_len]
    run3_texts = text_list[2*third_len:]
    
    run1_summaries = summary_list[:third_len]
    run2_summaries = summary_list[third_len:2*third_len]
    run3_summaries = summary_list[2*third_len:]

    query_types_run1 = query_types[:third_len]
    query_types_run2 = query_types[third_len:2*third_len]
    query_types_run3 = query_types[2*third_len:]

    run1_results = defaultdict(list)
    run2_results = defaultdict(list)
    run3_results = defaultdict(list)
    combined_results = defaultdict(list)

    def calculate_bertscore_for_runs(srcs_run1, tgts_run1, srcs_run2, tgts_run2, srcs_run3, tgts_run3):
        P1, R1, F1_run1 = score(srcs_run1, tgts_run1, model_type=model_path, verbose=True)
        P2, R2, F1_run2 = score(srcs_run2, tgts_run2, model_type=model_path, verbose=True)
        P3, R3, F1_run3 = score(srcs_run3, tgts_run3, model_type=model_path, verbose=True)

        F1_run1 = F1_run1.tolist()
        F1_run2 = F1_run2.tolist()
        F1_run3 = F1_run3.tolist()
        
        combined_F1 = [max(f1_r1, f1_r2, f1_r3) for f1_r1, f1_r2, f1_r3 in zip(F1_run1, F1_run2, F1_run3)]
        
        return F1_run1, F1_run2, F1_run3, combined_F1

    for category in set(query_types):
        run1_texts_cat = [text for text, qt in zip(run1_texts, query_types_run1) if qt == category]
        run1_summaries_cat = [summary for summary, qt in zip(run1_summaries, query_types_run1) if qt == category]

        run2_texts_cat = [text for text, qt in zip(run2_texts, query_types_run2) if qt == category]
        run2_summaries_cat = [summary for summary, qt in zip(run2_summaries, query_types_run2) if qt == category]

        run3_texts_cat = [text for text, qt in zip(run3_texts, query_types_run3) if qt == category]
        run3_summaries_cat = [summary for summary, qt in zip(run3_summaries, query_types_run3) if qt == category]

        F1_run1, F1_run2, F1_run3, combined_F1 = calculate_bertscore_for_runs(run1_texts_cat, run1_summaries_cat, run2_texts_cat, run2_summaries_cat, run3_texts_cat, run3_summaries_cat)

        run1_results[category] = F1_run1
        run2_results[category] = F1_run2
        run3_results[category] = F1_run3
        combined_results[category] = combined_F1

    overall_F1_run1 = sum(run1_results.values(), [])
    overall_F1_run2 = sum(run2_results.values(), [])
    overall_F1_run3 = sum(run3_results.values(), [])
    overall_combined_F1 = sum(combined_results.values(), [])

    def calculate_accuracy(f1_scores, threshold=0.57):
        correct_predictions = sum(1 for f1 in f1_scores if f1 >= threshold)
        accuracy = (correct_predictions / len(f1_scores)) * 100
        return accuracy

    run1_accuracy = calculate_accuracy(overall_F1_run1)
    run2_accuracy = calculate_accuracy(overall_F1_run2)
    run3_accuracy = calculate_accuracy(overall_F1_run3)
    combined_accuracy = calculate_accuracy(overall_combined_F1)

    print(f"Accuracy for Run 1 (Threshold 0.57): {run1_accuracy:.2f}%")
    print(f"Accuracy for Run 2 (Threshold 0.57): {run2_accuracy:.2f}%")
    print(f"Accuracy for Run 3 (Threshold 0.57): {run3_accuracy:.2f}%")
    print(f"Combined Accuracy (Threshold 0.57): {combined_accuracy:.2f}%")

    for category in set(query_types):
        avg_F1_cat_run1 = sum(run1_results[category]) / len(run1_results[category])
        avg_F1_cat_run2 = sum(run2_results[category]) / len(run2_results[category])
        avg_F1_cat_run3 = sum(run3_results[category]) / len(run3_results[category])
        avg_combined_F1_cat = sum(combined_results[category]) / len(combined_results[category])

        run1_cat_accuracy = calculate_accuracy(run1_results[category])
        run2_cat_accuracy = calculate_accuracy(run2_results[category])
        run3_cat_accuracy = calculate_accuracy(run3_results[category])
        combined_cat_accuracy = calculate_accuracy(combined_results[category])

        print(f"Category {category} - Run 1 Accuracy: {run1_cat_accuracy:.2f}%")
        print(f"Category {category} - Run 2 Accuracy: {run2_cat_accuracy:.2f}%")
        print(f"Category {category} - Run 3 Accuracy: {run3_cat_accuracy:.2f}%")
        print(f"Category {category} - Combined Accuracy: {combined_cat_accuracy:.2f}%")



def factcc_eval():
    model_path = 'manueldeprada/FactCC'

    file_path = os.path.join(_EVAL_DIR, 'eval_result')
    # gpt_output_file = f"{file_path}/factual_gpt_result.txt"
    output_file = f"{file_path}/factual_llama_baseline_result.txt"
    narrative_gt = f"{file_path}/factual_narrative_gt.txt"
    question_type_file = f"{file_path}/factual_gt.txt"  # Question types file

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    with open(output_file, 'r') as gpt_file, open(narrative_gt, 'r') as narrative_file, open(question_type_file, 'r') as type_file:
        text_list = gpt_file.readlines()
        summary_list = narrative_file.readlines()
        question_type_list = type_file.readlines()

    if len(text_list) != len(summary_list) or len(text_list) != len(question_type_list):
        raise ValueError("Mismatch between the number of lines in input files.")

    question_types = [int(q_type.strip()) for q_type in question_type_list]

    total_len = len(text_list)
    run_length = total_len // 3

    run1_text = text_list[:run_length]
    run1_summary = summary_list[:run_length]
    run2_text = text_list[run_length:2 * run_length]
    run2_summary = summary_list[run_length:2 * run_length]
    run3_text = text_list[2 * run_length:]
    run3_summary = summary_list[2 * run_length:]

    run1_types = question_types[:run_length]
    run2_types = question_types[run_length:2 * run_length]
    run3_types = question_types[2 * run_length:]

    results_by_type = {i: {"run_1_correct": 0, "run_2_correct": 0, "run_3_correct": 0, "combined_correct": 0, "total_1": 0, "total_2": 0, "total_3": 0, "total_pairs": 0} for i in set(question_types)}

    def evaluate_run(texts, summaries, q_types, run_id):
        correct_predictions = 0

        with open(f"{file_path}/factual_eval_run_{run_id}_results.txt", "w") as result_file:
            for text, summary, q_type in zip(texts, summaries, q_types):
                text = text.strip()
                summary = summary.strip()

                input_dict = tokenizer(text, summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
                logits = model(**input_dict).logits
                pred = logits.argmax(dim=1)
                result = model.config.id2label[pred.item()]

                if result == 'CORRECT':
                    correct_predictions += 1
                    if run_id == 1:
                        results_by_type[q_type]["run_1_correct"] += 1
                    elif run_id == 2:
                        results_by_type[q_type]["run_2_correct"] += 1
                    elif run_id == 3:
                        results_by_type[q_type]["run_3_correct"] += 1

                results_by_type[q_type][f"total_{run_id}"] += 1

                result_file.write(f"Text: {text}\nSummary: {summary}\nPrediction: {result}\n\n")

        return correct_predictions

    correct_1 = evaluate_run(run1_text, run1_summary, run1_types, run_id=1)
    correct_2 = evaluate_run(run2_text, run2_summary, run2_types, run_id=2)
    correct_3 = evaluate_run(run3_text, run3_summary, run3_types, run_id=3)

    total_1 = len(run1_text)
    total_2 = len(run2_text)
    total_3 = len(run3_text)

    print(f"Run 1 Accuracy@1: {(correct_1 / total_1) * 100:.2f}%")
    print(f"Run 2 Accuracy@1: {(correct_2 / total_2) * 100:.2f}%")
    print(f"Run 3 Accuracy@1: {(correct_3 / total_3) * 100:.2f}%")

    correct_at_2 = 0
    for text_1, summary_1, text_2, summary_2, text_3, summary_3, q_type in zip(run1_text, run1_summary, run2_text, run2_summary, run3_text, run3_summary, question_types):
        text_1, text_2, text_3 = text_1.strip(), text_2.strip(), text_3.strip()
        summary_1, summary_2, summary_3 = summary_1.strip(), summary_2.strip(), summary_3.strip()

        input_dict_1 = tokenizer(text_1, summary_1, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        logits_1 = model(**input_dict_1).logits
        pred_1 = logits_1.argmax(dim=1)
        result_1 = model.config.id2label[pred_1.item()]

        input_dict_2 = tokenizer(text_2, summary_2, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        logits_2 = model(**input_dict_2).logits
        pred_2 = logits_2.argmax(dim=1)
        result_2 = model.config.id2label[pred_2.item()]

        input_dict_3 = tokenizer(text_3, summary_3, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        logits_3 = model(**input_dict_3).logits
        pred_3 = logits_3.argmax(dim=1)
        result_3 = model.config.id2label[pred_3.item()]

        if result_1 == 'CORRECT' or result_2 == 'CORRECT' or result_3 == 'CORRECT':
            correct_at_2 += 1
            results_by_type[q_type]["combined_correct"] += 1

        results_by_type[q_type]["total_pairs"] += 1

    total_pairs = len(run1_text)
    acc2_percentage = (correct_at_2 / total_pairs) * 100

    print(f"Accuracy@3 (combined runs): {acc2_percentage:.2f}%")

    print("\nAccuracy by Question Type:")
    for q_type, data in results_by_type.items():
        run_1_acc = (data["run_1_correct"] / data["total_1"]) * 100 if data["total_1"] > 0 else 0
        run_2_acc = (data["run_2_correct"] / data["total_2"]) * 100 if data["total_2"] > 0 else 0
        run_3_acc = (data["run_3_correct"] / data["total_3"]) * 100 if data["total_3"] > 0 else 0
        combined_acc = (data["combined_correct"] / data["total_pairs"]) * 100 if data["total_pairs"] > 0 else 0

        print(f"Question Type {q_type}:")
        print(f"  Run 1 Accuracy@1: {run_1_acc:.2f}%")
        print(f"  Run 2 Accuracy@1: {run_2_acc:.2f}%")
        print(f"  Run 3 Accuracy@1: {run_3_acc:.2f}%")
        print(f"  Accuracy@3 (combined): {combined_acc:.2f}%")







def _compute_table3_metrics_for_one_method(text_list, summary_list, question_types, factcc_model, factcc_tokenizer, bertscore_model_name="bert-large-uncased", bertscore_threshold=0.57):
    # Set REPLICATE_SILENT_HANG=1 to reproduce original "no output after model load" behavior
    silent = os.environ.get("REPLICATE_SILENT_HANG", "").strip() == "1"
    n = len(text_list)
    run_len = n // 3
    run1_t = [t.strip() for t in text_list[:run_len]]
    run2_t = [t.strip() for t in text_list[run_len:2*run_len]]
    run3_t = [t.strip() for t in text_list[2*run_len:]]
    run1_s = [s.strip() for s in summary_list[:run_len]]
    run2_s = [s.strip() for s in summary_list[run_len:2*run_len]]
    run3_s = [s.strip() for s in summary_list[2*run_len:]]
    types1 = question_types[:run_len]

    def factcc_correct_one(text, summary):
        inp = factcc_tokenizer(text, summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
        pred = factcc_model(**inp).logits.argmax(dim=1).item()
        return factcc_model.config.id2label[pred] == 'CORRECT'

    if silent:
        factcc_run1 = [factcc_correct_one(t, s) for t, s in zip(run1_t, run1_s)]
        factcc_run2 = [factcc_correct_one(t, s) for t, s in zip(run2_t, run2_s)]
        factcc_run3 = [factcc_correct_one(t, s) for t, s in zip(run3_t, run3_s)]
    else:
        def _factcc_run(texts, summaries, run_label):
            out = []
            for i, (t, s) in enumerate(zip(texts, summaries)):
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  FactCC {run_label}: {i + 1}/{len(texts)}", flush=True)
                out.append(factcc_correct_one(t, s))
            return out
        factcc_run1 = _factcc_run(run1_t, run1_s, "run1")
        factcc_run2 = _factcc_run(run2_t, run2_s, "run2")
        factcc_run3 = _factcc_run(run3_t, run3_s, "run3")
    factcc_combined = [a or b or c for a, b, c in zip(factcc_run1, factcc_run2, factcc_run3)]

    _verbose = not silent
    if not silent:
        print("  BERTScore run1...", flush=True)
    _, _, F1_r1 = score(run1_t, run1_s, model_type=bertscore_model_name, verbose=_verbose)
    if not silent:
        print("  BERTScore run2...", flush=True)
    _, _, F1_r2 = score(run2_t, run2_s, model_type=bertscore_model_name, verbose=_verbose)
    if not silent:
        print("  BERTScore run3...", flush=True)
    _, _, F1_r3 = score(run3_t, run3_s, model_type=bertscore_model_name, verbose=_verbose)
    F1_r1, F1_r2, F1_r3 = F1_r1.tolist(), F1_r2.tolist(), F1_r3.tolist()
    bertscore_combined = [max(a, b, c) for a, b, c in zip(F1_r1, F1_r2, F1_r3)]

    def acc_at_threshold(scores, threshold=bertscore_threshold):
        return 100.0 * sum(1 for x in scores if x >= threshold) / len(scores) if scores else 0.0

    results = {}
    for cat_name, type_ids in TABLE3_CATEGORIES.items():
        mask = [t in type_ids for t in types1]
        n_cat = sum(mask)
        if n_cat == 0:
            results[cat_name] = (0.0, 0.0, 0.0, 0.0)
            continue
        f1 = 100.0 * sum(1 for i in range(run_len) if mask[i] and factcc_run1[i]) / n_cat
        f3 = 100.0 * sum(1 for i in range(run_len) if mask[i] and factcc_combined[i]) / n_cat
        b1_list = [F1_r1[i] for i in range(run_len) if mask[i]]
        b3_list = [bertscore_combined[i] for i in range(run_len) if mask[i]]
        b1 = acc_at_threshold(b1_list)
        b3 = acc_at_threshold(b3_list)
        results[cat_name] = (f1, f3, b1, b3)

    f1_overall = 100.0 * sum(factcc_run1) / run_len
    f3_overall = 100.0 * sum(factcc_combined) / run_len
    b1_overall = acc_at_threshold(F1_r1)
    b3_overall = acc_at_threshold(bertscore_combined)
    results["Overall"] = (f1_overall, f3_overall, b1_overall, b3_overall)
    return results


def run_table3_eval():
    # Default: eval_result next to this script (this repo). Override with EVAL_RESULT_DIR.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.environ.get("EVAL_RESULT_DIR")
    if file_path:
        file_path = os.path.expanduser(file_path)
    else:
        file_path = os.path.join(_script_dir, "eval_result")
    narrative_gt_file = os.path.join(file_path, "factual_narrative_gt.txt")
    question_type_file = os.path.join(file_path, "factual_gt.txt")

    with open(narrative_gt_file, 'r') as f:
        summary_list = [line.strip() for line in f.readlines()]
    with open(question_type_file, 'r') as f:
        question_types = [int(line.strip()) for line in f.readlines()]

    factcc_tokenizer = BertTokenizer.from_pretrained('manueldeprada/FactCC')
    factcc_model = BertForSequenceClassification.from_pretrained('manueldeprada/FactCC')
    silent = os.environ.get("REPLICATE_SILENT_HANG", "").strip() == "1"
    if not silent:
        print("FactCC model loaded. BERTScore will load on first use. Starting per-method evaluation (this can take a long time)...", flush=True)

    methods = [
        ("Llama3.1", os.path.join(file_path, "factual_llama_baseline_result.txt")),
        ("LogiEx-Llama", os.path.join(file_path, "factual_llama_result.txt")),
        ("GPT-4o", os.path.join(file_path, "vanilla_llm_result.txt")),
        ("LogiEx-GPT", os.path.join(file_path, "factual_gpt_result.txt")),
    ]

    all_results = {}
    for method_name, output_file in methods:
        if not silent:
            print(f"\nEvaluating method: {method_name}", flush=True)
        with open(output_file, 'r') as f:
            text_list = [line.strip() for line in f.readlines()]
        if len(text_list) != len(summary_list) or len(text_list) != len(question_types):
            raise ValueError(f"{method_name}: length mismatch pred={len(text_list)} gt={len(summary_list)} types={len(question_types)}")
        all_results[method_name] = _compute_table3_metrics_for_one_method(
            text_list, summary_list, question_types, factcc_model, factcc_tokenizer
        )
        if not silent:
            print(f"  Done: {method_name}", flush=True)

    cat_order = ["Base-Level", "Second-Level", "Logic Comparison", "Overall"]
    print("Table 3: Evaluation of LogiEx vs. baselines across evidence complexities. (BERTScore=B, FactCC=F)")
    print("Method          | Base-Level (F@1 F@3 B@1 B@3) | Second-Level (F@1 F@3 B@1 B@3) | Logic Comparison (F@1 F@3 B@1 B@3) | Overall (F@1 F@3 B@1 B@3)")
    print("-" * 140)
    for method_name in [m[0] for m in methods]:
        parts = [method_name.ljust(16)]
        for cat in cat_order:
            f1, f3, b1, b3 = all_results[method_name][cat]
            parts.append(" {:6.2f}% {:6.2f}% {:6.2f}% {:6.2f}% ".format(f1, f3, b1, b3))
        print("|".join(parts))
    print("-" * 140)


if __name__ == "__main__":
    run_table3_eval()

