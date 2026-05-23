import json
from tasks.transit_logics import *
from tasks.logic_parameterizer import *
import utils
import backend.tasks.embedding_utils as kb_utils
import parsimonious



file_path = "backend/data/transit_knowledge.md"
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.readlines()

# Cache for pre-computed knowledge base embeddings
_cached_content_lib = None

def _get_content_lib(client):
    """Return cached knowledge base embeddings, computing them once on first call."""
    global _cached_content_lib
    if _cached_content_lib is not None:
        return _cached_content_lib

    embedding_model = "text-embedding-3-small"
    print("Computing knowledge base embeddings (one-time)...")
    knowledge_encoding = [kb_utils.get_embedding(client, x, model=embedding_model) for x in content]
    content_lib = []
    for i in range(len(content)):
        content_lib_entry = {}
        content_lib_entry["text"] = content[i]
        content_lib_entry["embedding"] = knowledge_encoding[i]
        content_lib.append(content_lib_entry)
    _cached_content_lib = content_lib
    return _cached_content_lib


def simple_rag_qa(query, client, assistant, args):
    content_lib = _get_content_lib(client)

    context = ""
    strings, relatednesses = kb_utils.strings_ranked_by_relatedness(client, query, content_lib, top_n=3)
    for string, relatedness in zip(strings, relatednesses):
        if relatedness > 0.25:
            context += string

    if context == "":
        context = "I apologize. I do not have enough information to answer this question."

    print(context)
    user_prompt = "Please answer the question based on the context. Question: {}. Context: {}.".format(query, context)

    system_prompt = (
        "Your task is to respond to user inquiries about the Monte Carlo Tree Search (MCTS) algorithm's "
        "recommended actions for assigning vehicles in a paratransit service. When addressing these inquiries, "
        "you will: 1. Receive the user's question; 2. Receive a context for answering this question.\n\n"
        "Based on the above, you should provide clear, straightforward responses to the user about their "
        "QUESTIONS without any introductory and concluding remarks. Ensure your answers are easy to understand "
        "for non-technical users. DO NOT reply with a single integer.\n\n"
        "If the knowledge context is not clear or not relevant, just say you do not have enough information "
        "to answer this question."
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=args.temperature,
    )

    response_logs = []
    model_response = response.choices[0].message.content.strip()
    response_logs.append(model_response)

    return response_logs


def process_llm_answer(answers: str, data: json, scenario: int):
    processed_answers = [answer.replace("Answer: ", "").split("; ") for answer in answers] 

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
            print("Print a logic:",logic_fact)
            try:
                parsed = parse(logic_fact)
                score_function = scoring.get(type(parsed), lambda x, y: "Not applicable")
                result = score_function(parsed, data, scenario)
                print("Got result:", result)
                logic_checking_results.append(f"{logic_fact}: {result}")
            except parsimonious.exceptions.ParseError:
                print("ParseError")
    
    if len(logic_checking_results) == 0:
        return "ParseError"

    return "\n".join(logic_checking_results)



def generate_final_answer(args, client, assistant, prompt):
    system_prompt = (
        "Your task is to respond to user inquiries about the Monte Carlo Tree Search (MCTS) algorithm's "
        "recommended actions for assigning vehicles in a paratransit service. When addressing these inquiries, "
        "you will: 1. Receive the user's question; 2. Receive a logical expression that is relevant for "
        "providing an answer. 3. Receive the result of this logical expression's evaluation.\n\n"
        "Based on the above, you should provide clear, straightforward responses to the user without any "
        "introductory and concluding remarks. Ensure your answers are non-technical and AVOID any EXPLICIT "
        "mention of logic formulas and variable names. If the result from the logic evaluation contradicts "
        "the user's question, make sure to highlight this discrepancy.\n\n"
        "When explaining why a passenger was assigned to a vehicle, please verify the accuracy of the query. "
        "If the query is incorrect or contains inaccuracies, kindly point out the error and provide "
        "clarification. Note that the passenger will ALWAYS be assigned to the vehicle with a HIGHER reward.\n\n"
        "The following are meanings behind the logic expressions:\n"
        "- tp (request ID): Retrieves the designated pick-up time for a specific trip request.\n"
        "- td (request ID): Retrieves the designated drop-off time for a specific trip request.\n"
        "- C (vehicle ID): Retrieves the seating capacity of a vehicle identified by its ID.\n"
        "- O (node, vehicle ID): Obtains the current occupancy of a vehicle, identified by its ID, at a specific node.\n"
        "- sp (request ID, vehicle ID): Estimates the number of additional stops required to pick up a passenger, "
        "aside from the passenger's own stop, if the request is assigned to the specified vehicle.\n"
        "- sd (request ID, vehicle ID): Estimates the number of additional stops required to drop off a passenger, "
        "aside from the passenger's own stop, if the request is assigned to the specified vehicle.\n"
        "- r (vehicle ID): Represents the reward associated with assigning a specific vehicle.\n"
        "- rd1 (vehicle ID): Indicates the first component of the decomposed reward for a vehicle assignment, "
        "assessing the vehicle's adherence to timing requirements at a node.\n"
        "- rd2 (vehicle ID): Represents the second component of the decomposed reward for a vehicle assignment, "
        "suggesting that a higher reward correlates with the ability to serve more trips and reject fewer in the future.\n"
        "- eta (vehicle ID): Estimates the time of arrival if the current request is assigned to the specified vehicle.\n"
        "- viod(tp(request ID), eta(vehicle ID)): Represents the potential delay in pick-up or drop-off time "
        "a passenger might experience if assigned to the specified vehicle. Less delay is preferable.\n"
        "- vioa(tp(request ID), eta(vehicle ID)): Represents the potential advancement in pick-up or drop-off time "
        "a passenger might experience if assigned to the specified vehicle.\n"
        "- pctd(tp(request ID), eta(vehicle ID)): Represents the probability of a delay in pick-up or drop-off time "
        "a passenger might experience if assigned to the specified vehicle. Less percentage of delay is preferable.\n"
        "- pcta(tp(request ID), eta(vehicle ID)): Represents the probability of an advancement in pick-up or drop-off time "
        "a passenger might experience if assigned to the specified vehicle.\n"
        "- vcv (C (vehicle ID), O (node, vehicle ID)): Checks for capacity violations by comparing the capacity "
        "of a vehicle with its current occupancy at a specific node. A value of True indicates that the occupancy exceeds the capacity.\n"
        "- vcvq (C (vehicle ID), O (node, vehicle ID)): Calculates the difference between the capacity of a vehicle "
        "and its current occupancy, indicating how many more passengers can be accommodated.\n"
        "- Phi1 (vioa(tp(request ID 1), eta(vehicle ID 1)), vioa(tp(request ID 1), eta(vehicle ID 2))): "
        "Compares the timing violations of two potential vehicle assignments and returns a comparison result.\n"
        "- Phi2 (pcta(tp(request ID 1), eta(vehicle ID 1)), pcta(tp(request ID 1), eta(vehicle ID 2))): "
        "Compares the likelihood of timing violations for passengers in two potential vehicle assignments and returns a comparison result.\n"
        "- Phi3 (reward(vehicle ID 1), reward(vehicle ID 2)): Compares the rewards or decomposed rewards of two "
        "vehicle assignments by calculating the difference. A positive result indicates a higher reward for the first vehicle.\n"
        "- Phi4 (sp (request ID 1, vehicle ID 1), sp (request ID 2, vehicle ID 2)): Compares the number of "
        "unnecessary stops for two different vehicle assignments by subtracting the stops related to the second vehicle from the first.\n"
        "- search(vehicle ID) checks the possible consequences of assigning the trip to a vehicle that was not "
        "preferred initially. The returned values correspond to: advancements in pick-up time, delay in pick-up time, "
        "advancements in drop-off time, delay in drop-off time.\n"
        "- exclude(vehicle ID) checks the possible consequences when the specified vehicle breaks down. The first "
        "returned value represents the new vehicle assignment result. Other values correspond to: advancements in "
        "pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time.\n"
        "- Cong(request ID) checks the possible consequences in the event of a traffic jam. The returned values "
        "correspond to: advancements in pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time.\n"
        "- multi(number of passengers) checks the possible consequences of multiple passengers appearing in the same trip. "
        "The first returned value represents the new vehicle assignment result. Other values correspond to: advancements in "
        "pick-up time, delay in pick-up time, advancements in drop-off time, delay in drop-off time.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=args.temperature,
    )

    response_logs = []
    model_response = response.choices[0].message.content.strip()
    print("Explanation:", model_response)
    response_logs.append(model_response)

    return response_logs