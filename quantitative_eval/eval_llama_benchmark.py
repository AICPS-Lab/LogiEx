import argparse
import json
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
_BACKEND_ROOT = os.path.join(_PROJECT_ROOT, 'hscc-iccps26-paper254-repeatability-evaluations')
sys.path.insert(0, _BACKEND_ROOT)

import ollama
from eval_query_classify import fill_queries





def generate_final_answer(prompt): 

    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': "Your task is to respond to user inquiries about the Monte Carlo Tree Search (MCTS) algorithm's recommended actions for assigning vehicles in a paratransit service. When addressing these inquiries, you will: 1. Receive the user's question; 2. Receive a logical expression that is relevant for providing an answer. 3. Receive the result of this logical expression's evaluation. Based on the above, you should provide clear, straightforward responses to the user without any introductory and concluding remarks. Ensure your answers are non-technical and AVOID any EXPLICIT mention of logic formulas and variable names. If the result from the logic evaluation contradicts the user's question, make sure to highlight this discrepancy. When explaining why a passenger was assigned to a vehicle, please verify the accuracy of the query. If the query is incorrect or contains inaccuracies, kindly point out the error and provide clarification. Note that the passenger will ALWAYS be assigned to the vehicle with a HIGHER reward. The following are meanings behind the logic expressions: - tp (request ID): Retrieves the designated pick-up time for a specific trip request. - td (request ID): Retrieves the designated drop-off time for a specific trip request. - C (vehicle ID): Retrieves the seating capacity of a vehicle identified by its ID. - O (node, vehicle ID): Obtains the current occupancy of a vehicle, identified by its ID, at a specific node. - sp (request ID, vehicle ID): Estimates the number of additional stops required to pick up a passenger, aside from the passenger's own stop, if the request is assigned to the specified vehicle. - sd (request ID, vehicle ID): Estimates the number of additional stops required to drop off a passenger, aside from the passenger's own stop, if the request is assigned to the specified vehicle. - r (vehicle ID): Represents the reward associated with assigning a specific vehicle. - rd1 (vehicle ID): Indicates the first component of the decomposed reward for a vehicle assignment, assessing the vehicle's adherence to timing requirements at a node. - rd2 (vehicle ID): Represents the second component of the decomposed reward for a vehicle assignment, suggesting that a higher reward correlates with the ability to serve more trips and reject fewer in the future. - eta (vehicle ID): Estimates the time of arrival if the current request is assigned to the specified vehicle. - viod(tp(request ID), eta(vehicle ID)): Represents the potential delay in pick-up or drop-off time a passenger might experience if assigned to the specified vehicle. Less delay is preferable.  - vioa(tp(request ID), eta(vehicle ID)): Represents the potential advancement in pick-up or drop-off time a passenger might experience if assigned to the specified vehicle. - pctd(tp(request ID), eta(vehicle ID)): Represents the probability of a delay in pick-up or drop-off time a passenger might experience if assigned to the specified vehicle. Less percentage of delay is preferable.  - pcta(tp(request ID), eta(vehicle ID)): Represents the probability of an advancement in pick-up or drop-off time a passenger might experience if assigned to the specified vehicle. - vcv (C (vehicle ID), O (node, vehicle ID)): Checks for capacity violations by comparing the capacity of a vehicle with its current occupancy at a specific node. A value of True indicates that the occupancy exceeds the capacity. - vcvq (C (vehicle ID), O (node, vehicle ID)): Calculates the difference between the capacity of a vehicle and its current occupancy, indicating how many more passengers can be accommodated. - Phi1 (vioa(tp(request ID 1), eta(vehicle ID 1)), vioa(tp(request ID 1), eta(vehicle ID 2))): Compares the timing violations of two potential vehicle assignments and returns a comparison result. - Phi2 (pcta(tp(request ID 1), eta(vehicle ID 1)), pcta(tp(request ID 1), eta(vehicle ID 2))): Compares the likelihood of timing violations for passengers in two potential vehicle assignments and returns a comparison result. - Phi3 (reward(vehicle ID 1), reward(vehicle ID 2)): Compares the rewards or decomposed rewards of two vehicle assignments by calculating the difference. A positive result indicates a higher reward for the first vehicle. - Phi4 (sp (request ID 1, vehicle ID 1), sp (request ID 2, vehicle ID 2)): Compares the number of unnecessary stops for two different vehicle assignments by subtracting the stops related to the second vehicle from the first. - search(vehicle ID) checks the possible consequences of assigning the trip to a vehicle that was not preferred initially. The returned values correspond to: advancements in pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time.  - exclude(vehicle ID) checks the possible consequences when the specified vehicle breaks down. The first returned value represents the new vehicle assignment result. Other values correspond to: advancements in pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time.  - Cong(request ID) checks the possible consequences in the event of a traffic jam. The returned values correspond to: advancements in pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time. - multi(number of passengers) checks the possible consequences of multiple passengers appearing in the same trip. The first returned value represents the new vehicle assignment result. Other values correspond to: advancements in pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time. ",
        },
        {
            'role': 'user',
            'content': f"{prompt}",
        }
        ]
    )

    return response['message']['content']




def generate_baseline_final_answer(prompt): 

    response = ollama.chat(model='llama3.1', messages=
    [
        {
            'role': 'system',
            'content': "Your job is to clearly and directly explain the thought process of a Monte Carlo Tree Search algorithm by comparing different possible actions. The primary objective of the MCTS algorithm is to allocate requests to vehicles that optimize for the highest number of successfully met requests while simultaneously aiming to reduce timing violations. The uploaded JSON file starting with 'exp_eval' contains data stored within an MCTS search tree, specifically designed to assign paratransit vehicles based on user requests. Details of the file format: - decision epoch: A new request arrives in each decision epoch. The root node has a decision epoch of 0. - time: The time when the request was made. - current request: The requested pick-up and drop-off time of the request. - assign to: The vehicle assignment recommendation from MCTS. - eta: The estimated pick-up and drop-off times for the request under the current vehicle assignment, or 'Fulfilled' if already completed. - stops: The number of stops the request must wait for. - previous requests: The status of all previous requests. - vehicle status: The occupancy and capacity of all vehicles in the fleet. - N: The number of visits to this node. - R: The reward of this node. - decomposed R: Breaks down the reward into components, contributing to the total reward value. The higher the first value of decomposed reward is, the more trips are serviced; the higher the second value of decomposed reward is, the less the passenger needs to wait. - children: The possible subsequent nodes simulated from the current node's state, delineating the decision space. Each child node signifies a distinct outcome that arises from executing a particular action at the current node. Higher values of rewards, decomposed-reward, and visit-count are preferable, indicating a node's advantages. By default, the user is asking about the first decision epoch, which is decision epoch 0. You will provide direct answers about the information stored in the search tree. Please answer the questions you are given directly without any introductory or concluding statements. ",
        },
        {
            'role': 'user',
            'content': f"{prompt}",
        }
        ]
    )

    return response['message']['content']



def llama_baseline_answer_eval(
    search_tree_file=None,
    out_path=None,
):
    search_tree_file = search_tree_file or os.path.expanduser(
        os.path.join(_BACKEND_ROOT, 'backend', 'data', 'transit_eval_json', 'exp_eval_0.json')
    )
    out_path = out_path or os.path.join(_EVAL_DIR, 'eval_result')

    with open(search_tree_file, 'r') as file:
        data = json.load(file)

    query_list, gt_list, logic_gt, logic_results, narrative_gt = fill_queries()
    llama_result_final_answers = []

    for i, query in enumerate(query_list):
        print(i)
        print(query)
        try:
            prompt = "Content: " + str(data) + "\t"
            prompt += "Query: " + str(query)
            final_response = generate_baseline_final_answer(prompt)
            final_answer = final_response.replace('\n', ' ').strip()
            print(final_answer)
            llama_result_final_answers.append(final_answer)
        except Exception as e:
            print(f"The error: {e}")
            break

    out_file = os.path.join(out_path, 'factual_llama_baseline_result.txt')
    with open(out_file, 'a') as f:
        for line in llama_result_final_answers:
            f.write(str(line) + "\n")
    print(f"Wrote {len(llama_result_final_answers)} answers to {out_file}")




def llama_final_answer_eval(out_path=None):
    out_path = out_path or os.path.join(_EVAL_DIR, 'eval_result')

    query_list, gt_list, logic_gt, logic_results, narrative_gt = fill_queries()
    llama_result_final_answers = []

    for i, query in enumerate(query_list):
        print(i)
        try:
            gpt_reply = "Logic: " + logic_gt[i] + "\n"
            original_query = "Query: " + query + "\n"
            logic_checking_results = "Logic Checking Results: " + str(logic_results[i])
            final_response = generate_final_answer(
                original_query + gpt_reply + logic_checking_results
            )
            final_answer = final_response.replace('\n', ' ').strip()
            print(final_answer)
            llama_result_final_answers.append(final_answer)
        except Exception as e:
            print(f"The error: {e}")
            break

    out_file = os.path.join(out_path, 'factual_llama_result.txt')
    with open(out_file, 'a') as f:
        for line in llama_result_final_answers:
            f.write(str(line) + "\n")
    print(f"Wrote {len(llama_result_final_answers)} answers to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Llama 3.1 (Ollama) evaluation for MCTS paratransit Q&A: "
                    "baseline (JSON + query) or logic-assisted (query + logic + results)."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="baseline",
        choices=["baseline", "logic", "both"],
        help="Which evaluation to run: baseline (JSON+query), logic (query+logic+results), or both (default: baseline).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for result files (default: eval/eval_result).",
    )
    parser.add_argument(
        "--search-tree",
        default=None,
        help="Path to MCTS search tree JSON (only for baseline mode).",
    )
    args = parser.parse_args()

    out_path = os.path.expanduser(args.output_dir) if args.output_dir else None
    search_tree = os.path.expanduser(args.search_tree) if args.search_tree else None

    if args.mode in ("baseline", "both"):
        print("Running Llama baseline evaluation (JSON + query)...")
        llama_baseline_answer_eval(search_tree_file=search_tree, out_path=out_path)

    if args.mode in ("logic", "both"):
        print("Running Llama logic-assisted evaluation (query + logic + results)...")
        llama_final_answer_eval(out_path=out_path)


if __name__ == "__main__":
    main()