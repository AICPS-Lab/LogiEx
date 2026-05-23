import argparse
import os
import re
import sys
from openai import OpenAI
import ast
import random
import time
import json
from datetime import datetime
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, 'hscc-iccps26-paper254-repeatability-evaluations')
sys.path.insert(0, _BACKEND_ROOT)
import backend.prompts.test_prompting_transit as transit_logic_main 
import backend.prompts.transit_prompt as transit_prompts
from backend.src.transit.openai_calls import test_one_query 
import backend.prompts.transit_prompt as transit_prompt
import backend.utils as utils
from backend.tasks.transit_logics import *
from backend.tasks.logic_parameterizer import *
import backend.tasks.transit as transit_logic_task
import parsimonious

random.seed(16)


BASE_LEVEL = [1, 2, 3, 5, 6, 15, 30, 31] 
SECOND_LEVEL = [4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 25, 26, 27, 29] 
LOGIC_COMPARISON = [19, 20, 21, 28]  
TABLE4_CATEGORIES = {
    "Base-Level": BASE_LEVEL,
    "Second-Level": SECOND_LEVEL,
    "Logic Comparison": LOGIC_COMPARISON,
}


def _get_combined_correct_per_type_classification(pred_file, gt_file):
    with open(pred_file, 'r') as pred_f, open(gt_file, 'r') as gt_f:
        preds = [int(line.strip()) for line in pred_f.readlines()]
        gts = [int(line.strip()) for line in gt_f.readlines()]
    run_size = len(gts) // 3
    run1_preds = preds[:run_size]
    run2_preds = preds[run_size:2 * run_size]
    run3_preds = preds[2 * run_size:]
    run1_gts = gts[:run_size]
    query_type_correct_combined = {}
    query_type_counts = {}
    for i in range(run_size):
        gt = run1_gts[i]
        pred1, pred2, pred3 = run1_preds[i], run2_preds[i], run3_preds[i]
        if gt not in query_type_counts:
            query_type_counts[gt] = 0
            query_type_correct_combined[gt] = 0
        query_type_counts[gt] += 1
        if pred1 == gt or pred2 == gt or pred3 == gt:
            query_type_correct_combined[gt] += 1
    return query_type_correct_combined, query_type_counts


def _get_combined_correct_per_type_logic(pred_file, gt_file, logic_file):
    with open(pred_file, 'r') as pred_f, open(gt_file, 'r') as gt_f, open(logic_file, 'r') as lg_f:
        preds = [line.strip() for line in pred_f.readlines()]
        gts = [line.strip() for line in gt_f.readlines()]
        logics = [line.strip() for line in lg_f.readlines()]
    run_size = len(gts) // 3
    run1_preds = preds[:run_size]
    run2_preds = preds[run_size:2 * run_size]
    run3_preds = preds[2 * run_size:]
    run1_gts = gts[:run_size]
    run1_logics = logics[:run_size]

    def eval_logic_correctness(pred_lg, gt_lg):
        for j in range(len(pred_lg)):
            try:
                if parse(pred_lg[j]) != parse(gt_lg[j]):
                    return False
            except parsimonious.exceptions.ParseError:
                return False
        return True

    query_type_correct_combined = {}
    query_type_counts = {}
    for i in range(run_size):
        gt = int(run1_gts[i])
        logic_gt = run1_logics[i].split("; ")
        processed_1 = run1_preds[i].split("; ")
        processed_2 = run2_preds[i].split("; ")
        processed_3 = run3_preds[i].split("; ")
        if gt not in query_type_counts:
            query_type_counts[gt] = 0
            query_type_correct_combined[gt] = 0
        query_type_counts[gt] += 1
        if eval_logic_correctness(processed_1, logic_gt) or eval_logic_correctness(processed_2, logic_gt) or eval_logic_correctness(processed_3, logic_gt):
            query_type_correct_combined[gt] += 1
    return query_type_correct_combined, query_type_counts


def print_accuracy_by_table4_categories(pred_file, gt_file, logic_file=None):
    if logic_file is None:
        query_type_correct_combined, query_type_counts = _get_combined_correct_per_type_classification(pred_file, gt_file)
    else:
        query_type_correct_combined, query_type_counts = _get_combined_correct_per_type_logic(pred_file, gt_file, logic_file)
    # Normalize keys to int for category lookup
    correct = {int(k): v for k, v in query_type_correct_combined.items()}
    counts = {int(k): v for k, v in query_type_counts.items()}
    total_correct = 0
    total_count = 0
    lines = []
    for cat_name, type_ids in TABLE4_CATEGORIES.items():
        c = sum(correct.get(t, 0) for t in type_ids)
        n = sum(counts.get(t, 0) for t in type_ids)
        total_correct += c
        total_count += n
        pct = (c / n * 100) if n else 0
        lines.append((cat_name, pct, c, n))
    overall_pct = (total_correct / total_count * 100) if total_count else 0
    lines.append(("Overall", overall_pct, total_correct, total_count))
    for name, pct, c, n in lines:
        print(f"  {name}: {pct:.2f}% ({c}/{n})")
    return lines


def fill_placeholders(question, low_int=1, high_int=5):

    if question.count("{}") == 1:
        return question.format(random.randint(low_int, high_int))
    
    elif question.count("{}") == 2:
        num1, num2 = random.sample(range(low_int, high_int), 2)  # Ensure two different numbers
        return question.format(num1, num2)
    
    else:
        return question
    


def load_test_query_dataset(file_path, num_query=31, excluded_queries=[]):

    all_questions = []
    all_gt = []

    for type_id in range(1, num_query+1):

        if type_id not in excluded_queries:

            with open(os.path.join(file_path, 'query_{}.txt'.format(type_id)), 'r') as file:

                questions_list = [line.strip() for line in file.readlines() if line.strip()]

                gt_list = [type_id for i in range(len(questions_list))]

                filled_questions = [fill_placeholders(question, 0, 4) for question in questions_list]

                all_questions.extend(filled_questions)
                all_gt.extend(gt_list)

    combined = list(zip(all_questions, all_gt))
    random.shuffle(combined)
    x_shuffled, gt_shuffled = zip(*combined)
    x_shuffled = list(x_shuffled)
    gt_shuffled = list(gt_shuffled)

    return x_shuffled, gt_shuffled



def generate_testing_queries_llm(out_path, num_per_query, client):

    prompt_templates = {
        1: transit_prompts.standard_prompt_type_1,
        2: transit_prompts.standard_prompt_type_2,
        3: transit_prompts.standard_prompt_type_3,
        4: transit_prompts.standard_prompt_type_4,
        5: transit_prompts.standard_prompt_type_5,
        6: transit_prompts.standard_prompt_type_6,
        7: transit_prompts.standard_prompt_type_7,
        8: transit_prompts.standard_prompt_type_8,
        9: transit_prompts.standard_prompt_type_9,
        10: transit_prompts.standard_prompt_type_10,
        11: transit_prompts.standard_prompt_type_11,
        12: transit_prompts.standard_prompt_type_12,
        13: transit_prompts.standard_prompt_type_13,
        14: transit_prompts.standard_prompt_type_14,
        15: transit_prompts.standard_prompt_type_15,
        16: transit_prompts.standard_prompt_type_16,
        17: transit_prompts.standard_prompt_type_17,
        18: transit_prompts.standard_prompt_type_18,
        19: transit_prompts.standard_prompt_type_19,
        20: transit_prompts.standard_prompt_type_20,
        21: transit_prompts.standard_prompt_type_21,
        22: transit_prompts.standard_prompt_type_22,
        23: transit_prompts.standard_prompt_type_23,
        24: transit_prompts.standard_prompt_type_24,
        25: transit_prompts.standard_prompt_type_25,
        26: transit_prompts.standard_prompt_type_26,
        27: transit_prompts.standard_prompt_type_27,
        28: transit_prompts.standard_prompt_type_28,
        29: transit_prompts.standard_prompt_type_29,
        30: transit_prompts.standard_prompt_type_30,
        31: transit_prompts.standard_prompt_type_31,
    }

    for type_id in range(1, 32):

            prompt_template = prompt_templates.get(type_id)
            existing_queries = re.findall(r'Input: (.*?)\n', prompt_template)[:3]
            
            if type_id != 24: 
                existing_queries = [re.sub(r'vehicle \d+', 'vehicle {}', query) for query in existing_queries]
            else:
                existing_queries = [re.sub(r'\d+ passengers', '{} passengers', query) for query in existing_queries]

            all_variations = []

            for i in range(num_per_query):

                thread = client.beta.threads.create()

                assistant = client.beta.assistants.update("asst_Lul4NSVxtpEmbpRkjkQ7tiCy")

                str_queries = ", ".join(existing_queries)
                p_input = "Please generate 1 additional paraphrases of the following example queries without changing their meaning. Ensure that any braces {{}} in the examples are preserved in the generated paraphrases in the same format. The output should be in a list format, with each paraphrase on a new line, so it is easy to process and store in a text file later. Here are the example queries to paraphrase: {}".format(str_queries)

                message = client.beta.threads.messages.create(
                    thread_id=thread.id, role="user", content=p_input
                )

                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )

                utils.wait_on_run(run, client, thread)

                messages = client.beta.threads.messages.list(
                    thread_id=thread.id, order="asc", after=message.id
                )

                model_response = messages.data[0].content[0].text.value
                model_response = model_response.replace(';', ',')

                try: 
                    questions_list = ast.literal_eval(model_response) 
                    print("Generated:", questions_list)
                    all_variations.extend(questions_list)
                
                    if len(all_variations)>= num_per_query:
                        all_variations = all_variations[:num_per_query]
                        break

                except: 
                    pass

            with open(os.path.join(out_path, 'training_bert/query_{}.txt'.format(type_id)), "w") as file:
                for line in all_variations:
                    file.write(line + "\n")
    


def test_gpt_classify_acc():

    gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # DO NOT DELETE
    out_path = os.path.join(_EVAL_DIR, 'eval_data')
    generated_testing_data = generate_testing_queries_llm(
        out_path, 
        20, 
        gpt_client
    )

    ASSISTANT_ID = "asst_hsLdFmLHgwsrz0O7rqJKNVuF"
    gpt_assistant = gpt_client.beta.assistants.update(ASSISTANT_ID)

    file_path = os.path.join(_EVAL_DIR, 'eval_data')
    queries, gt = load_test_query_dataset(file_path, 31)

    gpt_result_types = []

    for i,query in enumerate(queries):
        
        try:
            query_type = transit_logic_main.llm_match_query(
                query, 
                gpt_client, 
                gpt_assistant
            )
            print("pred:", query_type, "\ttrue:", gt[i])
            gpt_result_types.append(query_type)
        except:
            break
        
        if i % 99 == 0:
            time.sleep(120)

    
    print(gpt_result_types)

    out_path = os.path.join(_EVAL_DIR, 'eval_result')
    with open(os.path.join(out_path, 'result_classify.txt'), "a") as file:
        for line in gpt_result_types:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'result_gt.txt'), "a") as file:
        for line in gt:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'result_query.txt'), "a") as file:
        for line in queries:
            file.write(line + "\n")



def get_logic_gt(query_type, value):

    conditions = {
        'type_1': f'tp(0)',
        'type_2': f'td(0)',
        'type_3': f'O(0,{value[0]})',
        'type_4': f'vcvq(C({value[0]}), O(1,{value[0]}))',
        'type_5': f'sp(0,{value[0]}); sd(0,{value[0]})',
        'type_6': f'eta({value[0]})',
        'type_7': f'viod(tp({value[0]}),eta({value[0]}))',
        'type_8': f'viod(td({value[0]}),eta({value[0]}))',
        'type_9': f'vioa(tp({value[0]}),eta({value[0]}))',
        'type_10': f'vioa(td({value[0]}),eta({value[0]}))',
        'type_11': f'pctd(tp({value[0]}),eta({value[0]}))',
        'type_12': f'pctd(td({value[0]}),eta({value[0]}))',
        'type_13': f'pcta(tp({value[0]}),eta({value[0]}))',
        'type_14': f'pcta(td({value[0]}),eta({value[0]}))',
        'type_15': f'sp(0,{value[0]}); sd(0,{value[0]})',
        'type_16': f'vcv(C({value[1]}), O(1,{value[1]})); Phi3(r({value[0]}), r({value[1]})); Phi3(rd1({value[0]}), rd1({value[1]})); Phi3(rd2({value[0]}), rd2({value[1]}))',
        'type_17': f'vcv(C({value[0]}), O(1,{value[0]}))',
        'type_18': f'search({value[0]})',
        'type_19': f'Phi1(vioa(tp({value[0]}),eta({value[0]})), vioa(tp({value[1]}),eta({value[1]}))); Phi1(vioa(td({value[0]}),eta({value[0]})), vioa(td({value[1]}),eta({value[1]}))); Phi1(viod(tp({value[0]}),eta({value[0]})), viod(tp({value[1]}),eta({value[1]}))); Phi1(viod(td({value[0]}),eta({value[0]})), viod(td({value[1]}),eta({value[1]}))); Phi4(sp(0,{value[0]}), sp(0,{value[1]})); Phi4(sd(0,{value[0]}), sd(0,{value[1]}))',
        'type_20': f'Phi1(vioa(tp({value[0]}),eta({value[0]})), vioa(tp({value[1]}),eta({value[1]}))); Phi1(vioa(td({value[0]}),eta({value[0]})), vioa(td({value[1]}),eta({value[1]}))); Phi1(viod(tp({value[0]}),eta({value[0]})), viod(tp({value[1]}),eta({value[1]}))); Phi1(viod(td({value[0]}),eta({value[0]})), viod(td({value[1]}),eta({value[1]})))',
        'type_21': f'Phi4(sp(0,{value[0]}), sp(0,{value[1]})); Phi4(sd(0,{value[0]}), sd(0,{value[1]}))',
        'type_22': f'Cong(0)',
        'type_23': f'exclude({value[0]})',
        'type_24': f'multi({value[0]})',
        'type_25': f'reassign({value[0]})',
        'type_26': f'viod(tp({value[0]}),eta({value[0]})); viod(td({value[0]}),eta({value[0]}))',
        'type_27': f'vioa(tp({value[0]}),eta({value[0]})); vioa(td({value[0]}),eta({value[0]}))',
        'type_28': f'Phi3(r({value[0]}), r({value[1]})); Phi3(rd1({value[0]}), rd1({value[1]})); Phi3(rd2({value[0]}), rd2({value[1]}))',
        'type_29': f'vcv(C({value[0]}), O(1,{value[0]})); r({value[0]}); rd1({value[0]}); rd2({value[0]})',
        'type_30': f'car(1)',
        'type_31': f'availablecar(1)',
    }

    result = conditions.get(f'type_{query_type}', f'default result {value}')
    
    return result



def process_logic(answers, data=None):
    
    processed_answers = [answer.split("; ") for answer in answers] 

    logic_checking_results = []
    scoring = {
        PickUpTime: quantitativescore,
        DropOffTime: quantitativescore,
        TreeVisit: quantitativescore, 
        Capacity: quantitativescore,
        Congestion: quantitativescore,
        MultiPass: quantitativescore,
        Exclude: quantitativescore,
        Reassign: quantitativescore,
        Occupancy: quantitativescore,
        StopsPickUp: quantitativescore, 
        StopsDropOff: quantitativescore, 
        Reward: quantitativescore, 
        DecompReward1: quantitativescore, 
        DecompReward2: quantitativescore, 
        ETA: quantitativescore,
        DegreeVioDelay: quantitativescore,
        DegreeVioAdv: quantitativescore,
        ChanceVioDelay: quantitativescore,
        ChanceVioAdv: quantitativescore,
        CapacityVio: qualitativescore,
        CapacityVioQuant: quantitativescore,
        CompVioTime: quantitativescore,
        CompPctTime: quantitativescore,
        CompReward: quantitativescore,
        CompNumStops: quantitativescore,
        AdditionalSearch: quantitativescore,
        CarAssign: quantitativescore,
        AvailableCar: quantitativescore,
    }

    for ans in processed_answers:
        for logic_fact in ans:
            try:
                parsed = parse(logic_fact)
                score_function = scoring.get(type(parsed), lambda x, y: "Not applicable")
                result = score_function(parsed, data, 0)
                logic_checking_results.append(f"{logic_fact}: {result}")
            except parsimonious.exceptions.ParseError:
                print("ParseError")
    
    if len(logic_checking_results) == 0:
        return "ParseError"

    return "\n".join(logic_checking_results)



def load_test_logic_dataset():

    file_path = os.path.join(_EVAL_DIR, 'eval_result')
    gt_file_n = "result_gt.txt"
    query_file_n = "result_query.txt"

    gt_file_path = os.path.join(file_path, gt_file_n)
    query_file_path = os.path.join(file_path, query_file_n)

    with open(gt_file_path, 'r') as file:
        gt_list = [int(line.strip()) for line in file.readlines()]
        gt_list = gt_list[:620]

    with open(query_file_path, 'r') as file:
        query_list = [line.strip() for line in file.readlines()]
        query_list = query_list[:620]

    file_path = os.path.join(_EVAL_DIR, 'eval_data')
    reference_queries = []
    for i in range(1, 32):
        ref_file_n = "query_{}.txt".format(i)
        ref_file_path = os.path.join(file_path, ref_file_n)
        with open(ref_file_path, 'r') as file:
            query_ref = [line.strip() for line in file.readlines()]
            reference_queries.append(query_ref[0])

    all_gt = []

    vehicle_pattern = r'vehicle (\d+)'
    passenger_pattern = r'(\d+) passengers'

    def extract_integers(query):
        vehicle_numbers = re.findall(vehicle_pattern, query)
        passenger_counts = re.findall(passenger_pattern, query) 
        return vehicle_numbers, passenger_counts
    
    for i in range(len(query_list)):
        query = query_list[i]
        vehicle_numbers, passenger_counts = extract_integers(query)

        if vehicle_numbers:
            assert (reference_queries[gt_list[i]-1].count("{}") == len(vehicle_numbers))
            vehicle_numbers = [int(item) for item in vehicle_numbers]
            if len(vehicle_numbers) == 1:
                vehicle_numbers.append(-1)
            all_gt.append(get_logic_gt(gt_list[i], vehicle_numbers))
            
        if passenger_counts:
            assert (reference_queries[gt_list[i]-1].count("{}") == len(passenger_counts))
            passenger_counts = [int(item) for item in passenger_counts]
            if len(passenger_counts) == 1:
                passenger_counts.append(-1)
            all_gt.append(get_logic_gt(gt_list[i], passenger_counts))

        if not vehicle_numbers and not passenger_counts:
            assert (reference_queries[gt_list[i]-1].count("{}") == 0)
            all_gt.append(get_logic_gt(gt_list[i], [-1,-1]))
    
    return query_list, gt_list, all_gt



def test_logic_generation_accuracy():

    query_list, gt_list, logic_gt = load_test_logic_dataset()
    
    args = None
    gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    ASSISTANT_ID = "asst_hsLdFmLHgwsrz0O7rqJKNVuF"
    gpt_assistant = gpt_client.beta.assistants.update(ASSISTANT_ID)

    gpt_result_types = []

    for i,query in enumerate(query_list):
        
        try:
            prompt = transit_logic_main.load_prompt(gt_list[i], query)

            gpt_reply = transit_logic_main.test_one_logic_prompt(args, gpt_client, gpt_assistant, prompt)

            print("pred:", gpt_reply[0].replace("Answer: ", ""), "\ttrue:", logic_gt[i])

            gpt_result_types.append(gpt_reply[0].replace("Answer: ", ""))
        
        except:
            break

        
        if i % 99 == 0:
            time.sleep(120)

    out_path = os.path.join(_EVAL_DIR, 'eval_result')
    with open(os.path.join(out_path, 'result_logic.txt'), "a") as file:
        for line in gpt_result_types:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'result_logic_gt.txt'), "a") as file:
        for line in logic_gt:
            file.write(str(line) + "\n")



def load_logic_dataset_from_list(query_list, gt_list):

    all_gt = []

    vehicle_pattern = r'vehicle (\d+)'
    passenger_pattern = r'(\d+) passengers'

    def extract_integers(query):
        vehicle_numbers = re.findall(vehicle_pattern, query)
        passenger_counts = re.findall(passenger_pattern, query) 
        return vehicle_numbers, passenger_counts
    
    for i in range(len(query_list)):
        query = query_list[i]
        vehicle_numbers, passenger_counts = extract_integers(query)

        if vehicle_numbers:
            vehicle_numbers = [int(item) for item in vehicle_numbers]
            if len(vehicle_numbers) == 1:
                vehicle_numbers.append(-1)
            all_gt.append(get_logic_gt(gt_list[i], vehicle_numbers))
            
        if passenger_counts:
            passenger_counts = [int(item) for item in passenger_counts]
            if len(passenger_counts) == 1:
                passenger_counts.append(-1)
            all_gt.append(get_logic_gt(gt_list[i], passenger_counts))

        if not vehicle_numbers and not passenger_counts:
            all_gt.append(get_logic_gt(gt_list[i], [-1,-1]))
    
    return query_list, gt_list, all_gt



def fill_queries():

    file_path = os.path.join(_EVAL_DIR, 'eval_data')
    queries, gt = load_test_query_dataset(
        file_path, 
        31, 
        excluded_queries=[18,22,23,24,25]
    )
    
    query_list, gt_list, logic_gt = load_logic_dataset_from_list(queries, gt)

    data_file = os.path.join(_BACKEND_ROOT, 'backend', 'data', 'transit_eval_json', 'exp_eval_0.json')
    with open(data_file, 'r') as file:
        data = json.load(file)

    logic_results = []
    narrative_gt = []

    for logic in logic_gt:
        logic_results.append(process_logic([logic], data))
        narrative_gt.append(generate_factual_consistency_gt([logic], data))

    return query_list, gt_list, logic_gt, logic_results, narrative_gt



def generate_factual_consistency_gt(answers, data=None):
    
    processed_answers = [answer.split("; ") for answer in answers] 

    scoring = {
        PickUpTime: quantitativescore,
        DropOffTime: quantitativescore,
        TreeVisit: quantitativescore, 
        Capacity: quantitativescore,
        Congestion: quantitativescore,
        MultiPass: quantitativescore,
        Exclude: quantitativescore,
        Reassign: quantitativescore,
        Occupancy: quantitativescore,
        StopsPickUp: quantitativescore, 
        StopsDropOff: quantitativescore, 
        Reward: quantitativescore, 
        DecompReward1: quantitativescore, 
        DecompReward2: quantitativescore, 
        ETA: quantitativescore,
        DegreeVioDelay: quantitativescore,
        DegreeVioAdv: quantitativescore,
        ChanceVioDelay: quantitativescore,
        ChanceVioAdv: quantitativescore,
        CapacityVio: qualitativescore,
        CapacityVioQuant: quantitativescore,
        CompVioTime: quantitativescore,
        CompPctTime: quantitativescore,
        CompReward: quantitativescore,
        CompNumStops: quantitativescore,
        AdditionalSearch: quantitativescore,
        CarAssign: quantitativescore,
        AvailableCar: quantitativescore,
    }

    final_narrative = ""

    for ans in processed_answers:
        for logic_fact in ans:
            try:
                parsed = parse(logic_fact)
                score_function = scoring.get(type(parsed), lambda x, y: "Not applicable")
                result = score_function(parsed, data, 0)
                final_narrative += load_narration(parsed, result)
                
            except parsimonious.exceptions.ParseError:
                print("ParseError")

    return final_narrative



def load_narration(parsed_logic, result):
    
    if isinstance(parsed_logic, DegreeVioAdv): 
        time_obj = datetime.strptime(result, '%H:%M')
        hours = time_obj.hour
        minutes = time_obj.minute

        vehicle_1 = int(parsed_logic.est_arrival.request)
        
        type_of_time = None
        if isinstance(parsed_logic.time, PickUpTime): 
            type_of_time = "pick-up time"
        elif isinstance(parsed_logic.time, DropOffTime): 
            type_of_time = "drop-off time"
        if hours > 0 or minutes > 0:
            return f"The {type_of_time} will be early by {hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''} if the passenger is assigned to vehicle {vehicle_1}. "
        else:
            return f"The {type_of_time} will not be earlier than the specified time if the passenger is assigned to vehicle {vehicle_1}. "

    elif isinstance(parsed_logic, DegreeVioDelay): 
        time_obj = datetime.strptime(result, '%H:%M')
        hours = time_obj.hour
        minutes = time_obj.minute

        vehicle_1 = int(parsed_logic.est_arrival.request)
        
        type_of_time = None
        if isinstance(parsed_logic.time, PickUpTime): 
            type_of_time = "pick-up time"
        elif isinstance(parsed_logic.time, DropOffTime): 
            type_of_time = "drop-off time"
        if hours > 0 or minutes > 0:
            return f"The {type_of_time} will be late by {hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''} if the passenger is assigned to vehicle {vehicle_1}. "
        else:
            return f"The {type_of_time} will not be later than the specified time if the passenger is assigned to vehicle {vehicle_1}. "

    elif isinstance(parsed_logic, ChanceVioDelay): 
        
        type_of_time = None
        if isinstance(parsed_logic.time, PickUpTime): 
            type_of_time = "picked up"
        elif isinstance(parsed_logic.time, DropOffTime): 
            type_of_time = "dropped off"
        
        return f"There is a {result} chance that the passenger will be {type_of_time} too late if assigned to vehicle {int(parsed_logic.est_arrival.request)}. "
        
    elif isinstance(parsed_logic, ChanceVioAdv): 
        
        type_of_time = None
        if isinstance(parsed_logic.time, PickUpTime): 
            type_of_time = "picked up"
        elif isinstance(parsed_logic.time, DropOffTime): 
            type_of_time = "dropped off"
        
        return f"There is a {result} chance that the passenger will be {type_of_time} too early if assigned to vehicle {int(parsed_logic.est_arrival.request)}. "

    elif isinstance(parsed_logic, CapacityVio): 
        if result == False:
            return f"There will not be a violation of the vehicle's capacity constraint for vehicle {int(parsed_logic.cap.vehicle)}. "
        elif result == True: 
            return f"There is a violation of the vehicle's capacity constraint for vehicle {int(parsed_logic.cap.vehicle)}. "

    elif isinstance(parsed_logic, CapacityVioQuant): 
        return f"Vehicle {int(parsed_logic.cap.vehicle)} can accommodate {int(result)} more passengers after assigning the current passenger. "
    
    elif isinstance(parsed_logic, CompVioTime): 
        type_of_time = None
        if isinstance(parsed_logic.left.time, PickUpTime): 
            type_of_time = "pick-up time"
        elif isinstance(parsed_logic.left.time, DropOffTime): 
            type_of_time = "drop-off time"

        type_of_check = None
        if isinstance(parsed_logic.left, DegreeVioDelay): 
            type_of_check = "delay"
        elif isinstance(parsed_logic.left, DegreeVioAdv): 
            type_of_check = "advancement"
        
        vehicle_1 = int(parsed_logic.left.est_arrival.request)
        vehicle_2 = int(parsed_logic.right.est_arrival.request)

        time_obj = datetime.strptime(result[0], '%H:%M')
        hours_1 = time_obj.hour
        minutes_1 = time_obj.minute

        time_obj = datetime.strptime(result[1], '%H:%M')
        hours_2 = time_obj.hour
        minutes_2 = time_obj.minute

        if hours_1 > hours_2 or (hours_1 == hours_2 and minutes_1 > minutes_2):
            comparison = f"Vehicle {vehicle_1} has a longer {type_of_check} for {type_of_time} compared to vehicle {vehicle_2}. "
        elif hours_1 < hours_2 or (hours_1 == hours_2 and minutes_1 < minutes_2):
            comparison = f"Vehicle {vehicle_1} has a shorter {type_of_check} for {type_of_time} compared to vehicle {vehicle_2}. "
        else:
            comparison = f"Vehicle {vehicle_1} and vehicle {vehicle_2} have the same {type_of_check} for {type_of_time}. "

        final_narrative = (f"For the {type_of_time}, vehicle {vehicle_1} has a {type_of_check} of {hours_1} hour{'s' if hours_1 != 1 else ''} "
                        f"and {minutes_1} minute{'s' if minutes_1 != 1 else ''}, while vehicle {vehicle_2} has a {type_of_check} of "
                        f"{hours_2} hour{'s' if hours_2 != 1 else ''} and {minutes_2} minute{'s' if minutes_2 != 1 else ''}. "
                        f"{comparison}")

        return final_narrative
    
    elif isinstance(parsed_logic, CompReward): 
        
        vehicle_1 = int(parsed_logic.left.node)
        vehicle_2 = int(parsed_logic.right.node)

        reward_type = None
        if isinstance(parsed_logic.left, Reward): 
            reward_type = "combine reward"
        elif isinstance(parsed_logic.left, DecompReward1): 
            reward_type = "reward for vehicle adherence to timing requirements"
        elif isinstance(parsed_logic.left, DecompReward2): 
            reward_type = "reward for the ability to serve more trips"

        if result > 0:
            comparison = f"Vehicle {vehicle_1} has a higher {reward_type} compared to vehicle {vehicle_2}. "
        elif result < 0:
            comparison = f"Vehicle {vehicle_1} has a lower {reward_type} compared to vehicle {vehicle_2}. "
        else:
            comparison = f"Vehicle {vehicle_1} and vehicle {vehicle_2} have the same {reward_type}. "

        final_narrative = (f"The {reward_type} for assigning the passenger to vehicle {vehicle_1} is compared with the {reward_type} for vehicle {vehicle_2}. "
                        f"The difference in reward is {abs(result):.2f}. {comparison}")

        return final_narrative
    

    elif isinstance(parsed_logic, CompNumStops): 
        
        vehicle_1 = int(parsed_logic.left.vehicle)
        vehicle_2 = int(parsed_logic.right.vehicle)

        stop_type = None
        if isinstance(parsed_logic.left, StopsPickUp): 
            stop_type = "pick up"
        elif isinstance(parsed_logic.left, StopsDropOff): 
            stop_type = "drop off"
        
        if result > 0:
            comparison = f"Vehicle {vehicle_1} requires more {stop_type} stops than vehicle {vehicle_2}. "
        elif result < 0:
            comparison = f"Vehicle {vehicle_1} requires fewer {stop_type} stops than vehicle {vehicle_2}. "
        else:
            comparison = f"Vehicle {vehicle_1} and vehicle {vehicle_2} require the same number of {stop_type} stops. "

        final_narrative = (f"The number of {stop_type} stops required for assigning the passenger to vehicle {vehicle_1} is compared with the stops required for vehicle {vehicle_2}. "
                        f"The difference in the number of stops is {abs(result)}. {comparison}")

        return final_narrative


    elif isinstance(parsed_logic, AvailableCar): 
        return f"There are {int(result)} vehicles available in total. "

    elif isinstance(parsed_logic, CarAssign): 
        return f"Vehicle {int(result)} has been assigned to fulfill the passenger's trip. "

    elif isinstance(parsed_logic, PickUpTime): 
        return f"The specified pick-up time is {result}. "

    elif isinstance(parsed_logic, DropOffTime): 
        return f"The specified drop-off time is {result}. "

    elif isinstance(parsed_logic, Capacity): 
        return f"The capacity of vehicle {int(parsed_logic.vehicle)} is {result}. "

    elif isinstance(parsed_logic, Occupancy): 
        return f"There are currently {int(result)} passengers onboard vehicle {parsed_logic.vehicle}. "

    elif isinstance(parsed_logic, StopsPickUp): 
        if int(result) == 0:
            return "There are no additional stops before picking up the passenger."
        else:
            return f"There are {int(result)} additional stops before picking up the passenger."

    elif isinstance(parsed_logic, StopsDropOff): 
        if int(result) == 0:
            return "There are no additional stops before dropping off the passenger."
        else:
            return f"There are {int(result)} additional stops before dropping off the passenger."

    elif isinstance(parsed_logic, Reward): 
        return f"The reward for assigning the passenger to vehicle {int(parsed_logic.node)} is {result}. "

    elif isinstance(parsed_logic, DecompReward1): 
        return f"The reward for vehicle adherence to timing requirements when assigning the passenger to vehicle {int(parsed_logic.node)} is {result}. "

    elif isinstance(parsed_logic, DecompReward2): 
        return f"The reward for the ability to serve more trips by assigning the passenger to vehicle {int(parsed_logic.node)} is {result}. "

    elif isinstance(parsed_logic, ETA): 
        return f"The estimated time of arrival for the passenger assigned to vehicle {int(parsed_logic.request)} is {result}. "



def temp_file_process():
    with open('eval/eval_result/note.txt', 'r') as file:
        lines = file.readlines()

    final_explanations = []
    current_explanation = []

    for line in lines:
        line = line.strip()
        
        if line.isdigit():

            if current_explanation:
                final_explanations.append(' '.join(current_explanation))
                current_explanation = [] 
        else:

            if line.startswith("Explanation: "):
                line = line.replace("Explanation: ", "")
            
            current_explanation.append(line)

    if current_explanation:
        final_explanations.append(' '.join(current_explanation))

    with open('eval/eval_result/cleaned_explanations.txt', 'w') as file:
        for explanation in final_explanations:
            file.write(explanation + '\n')

    print("Explanations have been cleaned and saved.")
        


def eval_final_response():
    
    query_list, gt_list, logic_gt, logic_results, narrative_gt = fill_queries()

    args = None
    gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    ASSISTANT_ID = "asst_Lalq7A6sjC0qo7Rmb9PTMaak"
    gpt_assistant = gpt_client.beta.assistants.update(ASSISTANT_ID)

    gpt_result_final_answers = []

    for i,query in enumerate(query_list):
        
        print(i)

        try:
            gpt_reply = "Logic: " + logic_gt[i] + "\n"
            original_query = "Query: " + query + "\n"
            logic_checking_results = "Logic Checking Results: " + str(logic_results[i])

            final_response = transit_logic_task.generate_final_answer(
                            args, gpt_client, gpt_assistant, 
                            original_query+gpt_reply+logic_checking_results)

            final_answer = final_response[0].replace('\n', ' ').strip()

            gpt_result_final_answers.append(final_answer)

        except Exception as e:
            print(f"The error: {e}")
            break

        if i % 49 == 0:
            time.sleep(100)

    out_path = os.path.join(_EVAL_DIR, 'eval_result')
    with open(os.path.join(out_path, 'factual_queries.txt'), "a") as file:
        for line in query_list:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'factual_gt.txt'), "a") as file:
        for line in gt_list:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'factual_logic_gt.txt'), "a") as file:
        for line in logic_gt:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'factual_logic_results.txt'), "a") as file:
        for line in logic_results:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'factual_narrative_gt.txt'), "a") as file:
        for line in narrative_gt:
            file.write(str(line) + "\n")

    with open(os.path.join(out_path, 'factual_gpt_result.txt'), "a") as file:
        for line in gpt_result_final_answers:
            file.write(str(line) + "\n")



def eval_vanilla_llm_final_response():
    search_tree_file = os.path.join(_BACKEND_ROOT, 'backend', 'data', 'transit_eval_json', 'exp_eval_0.json')

    gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    vector_store = gpt_client.vector_stores.create(
        name="Search Tree Reference"
    )
    uploaded_file = gpt_client.files.create(
        file=open(search_tree_file, "rb"),
        purpose="assistants"
    )
    
    vector_store_file = gpt_client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id
    )

    gpt_assistant = gpt_client.beta.assistants.update(
        assistant_id='asst_54Knb190gRj6sJSdfIjXyb3g',
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    print("Uploaded file ID:", uploaded_file.id)
    print("Vector_store ID:", vector_store.id)
    
    vector_store_id = vector_store.id
    upload_file_id = uploaded_file.id

    query_list, gt_list, logic_gt, logic_results, narrative_gt = fill_queries()

    vanilla_result_final_answers = []
    
    for i,query in enumerate(query_list):

        try:
            gpt_reply = test_one_query(
                1, 
                gpt_client, 
                gpt_assistant, 
                query
            )

            gpt_reply = gpt_reply[0].replace('\n', ' ').strip()

            print(gpt_reply)

            vanilla_result_final_answers.append(gpt_reply)

        except Exception as e:
            print(f"The error: {e}")
            break

        if i % 49 == 0:
            time.sleep(120)

    out_path = os.path.join(_EVAL_DIR, 'eval_result')
    
    with open(os.path.join(out_path, 'vanilla_llm_result.txt'), "a") as file:
        for line in vanilla_result_final_answers:
            file.write(str(line) + "\n")


def calculate_accuracy_three_runs(pred_file, gt_file, acc_by_type=False):
    with open(pred_file, 'r') as pred_f, open(gt_file, 'r') as gt_f:
        preds = [int(line.strip()) for line in pred_f.readlines()]
        gts = [int(line.strip()) for line in gt_f.readlines()]

    run_size = len(gts) // 3
    run1_preds = preds[:run_size]
    run2_preds = preds[run_size:2 * run_size]
    run3_preds = preds[2 * run_size:]
    
    run1_gts = gts[:run_size]
    run2_gts = gts[run_size:2 * run_size]
    run3_gts = gts[2 * run_size:]

    correct_run1 = 0
    correct_run2 = 0
    correct_run3 = 0
    combined_correct = 0

    query_type_counts = {}
    query_type_correct_run1 = {}
    query_type_correct_run2 = {}
    query_type_correct_run3 = {}
    query_type_correct_combined = {}

    for i in range(run_size):
        gt = run1_gts[i]
        pred1 = run1_preds[i]
        pred2 = run2_preds[i]
        pred3 = run3_preds[i]

        if gt not in query_type_counts:
            query_type_counts[gt] = 0
            query_type_correct_run1[gt] = 0
            query_type_correct_run2[gt] = 0
            query_type_correct_run3[gt] = 0
            query_type_correct_combined[gt] = 0
        query_type_counts[gt] += 1

        if pred1 == gt:
            correct_run1 += 1
            query_type_correct_run1[gt] += 1
        if pred2 == gt:
            correct_run2 += 1
            query_type_correct_run2[gt] += 1
        if pred3 == gt:
            correct_run3 += 1
            query_type_correct_run3[gt] += 1

        if pred1 == gt or pred2 == gt or pred3 == gt:
            combined_correct += 1
            query_type_correct_combined[gt] += 1

    total = run_size
    overall_accuracy_run1 = correct_run1 / total
    overall_accuracy_run2 = correct_run2 / total
    overall_accuracy_run3 = correct_run3 / total
    overall_accuracy_combined = combined_correct / total

    print(f"Run 1 Accuracy: {overall_accuracy_run1 * 100:.2f}%")
    print(f"Run 2 Accuracy: {overall_accuracy_run2 * 100:.2f}%")
    print(f"Run 3 Accuracy: {overall_accuracy_run3 * 100:.2f}%")
    print(f"Combined Accuracy: {overall_accuracy_combined * 100:.2f}%")

    if acc_by_type:
        print("\nAccuracy by Query Type (combined):")
        for query_type, count in sorted(query_type_counts.items()):
            acc_run1 = query_type_correct_run1[query_type] / count
            acc_run2 = query_type_correct_run2[query_type] / count
            acc_run3 = query_type_correct_run3[query_type] / count
            acc_combined = query_type_correct_combined[query_type] / count
            # print(f"Query Type {query_type}:")
            # print(f"  Run 1: {acc_run1 * 100:.2f}% ({query_type_correct_run1[query_type]}/{count})")
            # print(f"  Run 2: {acc_run2 * 100:.2f}% ({query_type_correct_run2[query_type]}/{count})")
            # print(f"  Run 3: {acc_run3 * 100:.2f}% ({query_type_correct_run3[query_type]}/{count})")
            print(f"Query Type {query_type}: Combined {acc_combined * 100:.2f}% ({query_type_correct_combined[query_type]}/{count})")



def calculate_logic_accuracy(pred_file, gt_file, logic_file, acc_by_type=False):
    with open(pred_file, 'r') as pred_f, open(gt_file, 'r') as gt_f, open(logic_file, 'r') as lg_f:
        preds = [line.strip() for line in pred_f.readlines()]
        gts = [line.strip() for line in gt_f.readlines()]
        logics = [line.strip() for line in lg_f.readlines()]
    
    run_size = len(gts) // 3
    run1_preds = preds[:run_size]
    run2_preds = preds[run_size:2 * run_size]
    run3_preds = preds[2 * run_size:]

    run1_logics = logics[:run_size]
    run1_gts = gts[:run_size]

    correct_run1 = 0
    correct_run2 = 0
    correct_run3 = 0
    combined_correct = 0

    query_type_counts = {}
    query_type_correct_run1 = {}
    query_type_correct_run2 = {}
    query_type_correct_run3 = {}
    query_type_correct_combined = {}

    def eval_logic_correctness(pred_lg, gt_lg):
        for i in range(len(pred_lg)):
            try:
                if parse(pred_lg[i]) != parse(gt_lg[i]):
                    # print("ERROR:", pred_lg[i], gt_lg[i])
                    return False
            except parsimonious.exceptions.ParseError:
                print("ParseError", pred_lg, gt_lg)
        return True

    for i in range(run_size):
        gt = run1_gts[i]
        logic_gt = run1_logics[i].split("; ")
        pred1 = run1_preds[i]
        pred2 = run2_preds[i]
        pred3 = run3_preds[i]

        processed_answers_1 = pred1.split("; ")
        processed_answers_2 = pred2.split("; ")
        processed_answers_3 = pred3.split("; ") 

        if gt not in query_type_counts:
            query_type_counts[gt] = 0
            query_type_correct_run1[gt] = 0
            query_type_correct_run2[gt] = 0
            query_type_correct_run3[gt] = 0
            query_type_correct_combined[gt] = 0
        query_type_counts[gt] += 1
        
        result_1 = eval_logic_correctness(processed_answers_1,logic_gt)
        result_2 = eval_logic_correctness(processed_answers_2,logic_gt)
        result_3 = eval_logic_correctness(processed_answers_3,logic_gt)

        if result_1 == True:
            correct_run1 += 1
            query_type_correct_run1[gt] += 1
        if result_2 == True:
            correct_run2 += 1
            query_type_correct_run2[gt] += 1
        if result_3 == True:
            correct_run3 += 1
            query_type_correct_run3[gt] += 1

        if result_1 == True or result_2 == True or result_3 == True:
            combined_correct += 1
            query_type_correct_combined[gt] += 1

    total = run_size
    overall_accuracy_run1 = correct_run1 / total
    overall_accuracy_run2 = correct_run2 / total
    overall_accuracy_run3 = correct_run3 / total
    overall_accuracy_combined = combined_correct / total

    print(f"Run 1 Accuracy: {overall_accuracy_run1 * 100:.2f}%")
    print(f"Run 2 Accuracy: {overall_accuracy_run2 * 100:.2f}%")
    print(f"Run 3 Accuracy: {overall_accuracy_run3 * 100:.2f}%")
    print(f"Combined Accuracy: {overall_accuracy_combined * 100:.2f}%")

    if acc_by_type:
        print("\nLogic accuracy by Query Type (combined):")
        for query_type, count in sorted(query_type_counts.items(), key=lambda x: int(x[0])):
            acc_run1 = query_type_correct_run1[query_type] / count
            acc_run2 = query_type_correct_run2[query_type] / count
            acc_run3 = query_type_correct_run3[query_type] / count
            acc_combined = query_type_correct_combined[query_type] / count
            # print(f"Query Type {query_type}:")
            # print(f"  Run 1: {acc_run1 * 100:.2f}% ({query_type_correct_run1[query_type]}/{count})")
            # print(f"  Run 2: {acc_run2 * 100:.2f}% ({query_type_correct_run2[query_type]}/{count})")
            # print(f"  Run 3: {acc_run3 * 100:.2f}% ({query_type_correct_run3[query_type]}/{count})")
            print(f"Query Type {query_type}: Combined {acc_combined * 100:.2f}% ({query_type_correct_combined[query_type]}/{count})")
    #     print(f"  Combined: {acc_combined * 100:.2f}% ({query_type_correct_combined[query_type]}/{count})")




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query classification and logic generation (BERT, LogiEx), "
                    "or regenerate vanilla LLM final responses."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="accuracy",
        choices=["accuracy", "regenerate", "both"],
        help="Run accuracy evaluation, regenerate final responses, or both (default: accuracy).",
    )
    parser.add_argument(
        "--acc-by-type",
        action="store_true",
        help="Print per-query-type (and Table 4 category) accuracy.",
    )
    args = parser.parse_args()

    result_dir = "eval/eval_result"
    gt_file = os.path.join(result_dir, "result_gt.txt")
    acc_by_type = args.acc_by_type

    if args.mode in ("regenerate", "both"):
        eval_final_response()

    if args.mode in ("accuracy", "both"):
        pred_file = os.path.join(result_dir, "bert_result.txt")
        print("================================================")
        print("BERT query classification accuracy calculation:")
        calculate_accuracy_three_runs(pred_file, gt_file, acc_by_type=acc_by_type)
        print("Table 4 categories (BERT classification):")
        print_accuracy_by_table4_categories(pred_file, gt_file)

        pred_file = os.path.join(result_dir, "result_classify.txt")
        print("================================================")
        print("LogiEx query classification accuracy calculation:")
        calculate_accuracy_three_runs(pred_file, gt_file, acc_by_type=acc_by_type)
        print("Table 4 categories (LogiEx classification):")
        print_accuracy_by_table4_categories(pred_file, gt_file)

        logic_file = os.path.join(result_dir, "result_logic_gt.txt")
        pred_file = os.path.join(result_dir, "result_logic.txt")
        print("================================================")
        print("LogiEx logic generation accuracy calculation:")
        calculate_logic_accuracy(pred_file, gt_file, logic_file, acc_by_type=acc_by_type)
        print("Table 4 categories (LogiEx logic generation):")
        print_accuracy_by_table4_categories(pred_file, gt_file, logic_file=logic_file)


if __name__ == "__main__":
    main()