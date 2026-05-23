
import os
import sys
import re
import random 
import backend.prompts.transit_prompt as transit_prompts
module_dir = '../queries'
current_dir = os.path.dirname(__file__)  
path_to_module = os.path.join(current_dir, module_dir)
sys.path.insert(0, path_to_module)
import transit_queries_by_type as queries
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import utils



def extract_vehicle_numbers(query):
    """Extract all vehicle numbers mentioned in a query."""
    return re.findall(r'vehicle (\d+)', query)


def llm_match_query(query, client, assistant, args=None):
    """Using Chat Completions API for fast query classification."""

    system_prompt = (
        "You are a query classifier for a paratransit route planning system. "
        "Your task is to classify a user query into exactly one of the numbered categories below.\n\n"
        "RULES:\n"
        "- Respond with ONLY a single integer (the category number).\n"
        "- Do NOT output any explanation, punctuation, or extra text.\n"
        "- Match based on the semantic intent of the query, not exact wording. "
        "Users may phrase the same question in many different ways.\n"
        "- In the category examples below, {} is a placeholder for an actual vehicle number (e.g., \"vehicle 3\"). "
        "The user's query will contain real numbers instead of {}.\n"
        "- Categories 1, 2, 22, 30, and 31 are general questions that do NOT require a vehicle number.\n"
        "- All other categories require at least one vehicle number. "
        "If a query seems to fit one of those categories but does not mention any vehicle, return -1.\n"
        "- If the query does not fit any category, return -1.\n\n"
        "CATEGORIES:\n\n"
        "--- Passenger schedule ---\n"
        "1. Passenger pick-up time (e.g., \"Could you provide the designated pick-up time for the passenger?\")\n"
        "2. Passenger drop-off time (e.g., \"Could you tell me the planned drop-off time for the passenger?\")\n\n"
        "--- Vehicle status (single vehicle) ---\n"
        "3. Current passenger count on a vehicle (e.g., \"How many passengers are currently in vehicle {}?\")\n"
        "4. Remaining seating capacity of a vehicle (e.g., \"What is the current available seating capacity for vehicle {}?\")\n"
        "5. Number of stops before destination for a vehicle (e.g., \"What is the number of stops a passenger would encounter if assigned to vehicle {}?\")\n"
        "6. Estimated time of arrival for a vehicle (e.g., \"Could you provide the ETA for the passenger if assigned to vehicle {}?\")\n\n"
        "--- Time violations (single vehicle) ---\n"
        "7. Expected delay in PICK-UP time (e.g., \"How long is the delay expected to be in the pick-up time when assigned to vehicle {}?\")\n"
        "8. Expected delay in DROP-OFF time (e.g., \"How long is the delay expected to be in the drop-off time when assigned to vehicle {}?\")\n"
        "9. Expected advancement in PICK-UP time (e.g., \"How long is the advancement expected to take in the pick-up time when assigned to vehicle {}?\")\n"
        "10. Expected advancement in DROP-OFF time (e.g., \"How long is the advancement expected to take in the drop-off time when assigned to vehicle {}?\")\n"
        "11. Probability of PICK-UP delay (e.g., \"What are the chances the passenger will be delayed in their pick-up time when assigned to vehicle {}?\")\n"
        "12. Probability of DROP-OFF delay (e.g., \"What are the chances the passenger will be delayed in their drop-off time when assigned to vehicle {}?\")\n"
        "13. Probability of early PICK-UP (e.g., \"How likely is it that the passenger will be picked up ahead of schedule in vehicle {}?\")\n"
        "14. Probability of early DROP-OFF / arrival (e.g., \"What are the chances the passenger will arrive early if assigned to vehicle {}?\")\n"
        "15. Cause of delay for a vehicle (e.g., \"What causes the delay when assigning a passenger to vehicle {}?\")\n\n"
        "--- Vehicle comparisons (two vehicles) ---\n"
        "16. Why one vehicle was favored over another (e.g., \"Why was vehicle {} favored instead of vehicle {}?\")\n"
        "17. Why an alternative vehicle was not considered (e.g., \"Why did the algorithm not consider alternative vehicle {}?\")\n"
        "18. Consequences of assigning to an alternative vehicle (e.g., \"What could result from assigning the passenger to alternative vehicle {}?\")\n"
        "19. Benefits of one vehicle over another ignoring capacity (e.g., \"What benefits does vehicle {} offer over vehicle {} when capacity isn't an issue?\")\n"
        "20. Comparing time violation reduction between vehicles (e.g., \"Does vehicle {} reduce time violations more effectively than vehicle {}?\")\n"
        "21. Comparing number of stops between vehicles (e.g., \"Is the number of stops lower for passengers in vehicle {} than in vehicle {}?\")\n\n"
        "--- Scenario / contingency ---\n"
        "22. What happens when traffic is congested (e.g., \"What does occur when traffic becomes congested?\")\n"
        "23. What happens if a vehicle becomes inoperable (e.g., \"What will happen if vehicle {} becomes inoperable?\")\n"
        "24. Handling a trip with multiple passengers (e.g., \"How do we handle this trip if it carries {} passengers?\")\n"
        "25. Which vehicle takes over if a vehicle breaks down (e.g., \"Which vehicle will take over the passengers from vehicle {} if it breaks down?\")\n\n"
        "--- Delay / early yes-no ---\n"
        "26. Will the passenger experience a delay in a vehicle (e.g., \"Will the passenger experience a delay if assigned to vehicle {}?\")\n"
        "27. Could the passenger arrive too early in a vehicle (e.g., \"Could the passenger be too early if they are assigned to vehicle {}?\")\n\n"
        "--- Reward / assignment ---\n"
        "28. Reward comparison between two vehicles (e.g., \"How do the rewards compare for assigning the passenger to vehicle {} compared to vehicle {}?\")\n"
        "29. Why the system assigned the passenger to a vehicle (e.g., \"Why did the system assign the passenger to vehicle {}?\")\n"
        "30. Which vehicle is assigned to the passenger (e.g., \"Which vehicle is assigned to fulfill the passenger's request?\")\n"
        "31. How many vehicles are available (e.g., \"How many vehicles are available to pick up the passenger?\")\n"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0,
        max_tokens=5,
    )

    model_response = response.choices[0].message.content.strip()

    try:
        return int(model_response)
    except ValueError:
        return -1


def match_query(new_query, vehicle_numbers):
    """Attempt to match the query against known templates."""

    if not vehicle_numbers:
        for template in queries.type_1:
            if template.format('') == new_query: 
                return 1
        for template in queries.type_2:
            if template.format('') == new_query:
                return 2
        return -1

    if len(vehicle_numbers) == 1:
        for template in queries.type_3:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 3
        for template in queries.type_4:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 4
        for template in queries.type_5:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 5
        for template in queries.type_6:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 6
        for template in queries.type_7:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 7
        for template in queries.type_8:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 8
        for template in queries.type_9:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 9
        for template in queries.type_10:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 10
        for template in queries.type_11:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 11
        for template in queries.type_13:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 13
        for template in queries.type_14:
            for number in vehicle_numbers:
                if template.format(number) == new_query:
                    return 14
                
    elif len(vehicle_numbers) == 2:
        for template in queries.type_12:
            if template.format(*vehicle_numbers) == new_query:
                return 12
        for template in queries.type_15:
            if template.format(*vehicle_numbers) == new_query:
                return 15
        for template in queries.type_16:
            if template.format(*vehicle_numbers) == new_query:
                return 16
        for template in queries.type_17:
            if template.format(*vehicle_numbers) == new_query:
                return 17

    return -1




def fill_and_load_queries(type_id, data=None):
    if type_id == 12:
        best_node_vehicle = 0
        best_node_vehicle_id = 0
        node_children = data["children"]
        for i in range(len(node_children)):
            if float(node_children[i]["R"]) > float(node_children[best_node_vehicle_id]["R"]):
                best_node_vehicle = int(node_children[i]["assign to"][4:])
                best_node_vehicle_id = i

    max_vehicle_number = len(data["children"])-1
    raw_queries = queries.load_query_by_type(type_id)
    filled_questions = []
    for question in raw_queries:
        if type_id == 12:
            vehicle1 = best_node_vehicle
        else:
            vehicle1 = random.randint(1, max_vehicle_number)
        vehicle2 = random.randint(1, max_vehicle_number)
        while vehicle2 == vehicle1:
            vehicle2 = random.randint(1, max_vehicle_number)
        random_float = random.random()
        if random_float > 0.8: 
            vehicle1, vehicle2 = vehicle2, vehicle1
        filled_question = question.format(vehicle1, vehicle2)
        filled_questions.append(filled_question)
    return filled_questions



def load_prompt(type_id, new_input):
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
    prompt_template = prompt_templates.get(type_id)
    if prompt_template:
        return prompt_template.format(input=new_input)
    else:
        return "Prompt template for type {} not found.".format(type_id)
    


def test_one_logic_prompt(args, client, assistant, prompt):
    system_prompt = (
        "Given the following pattern of questions and answers, generate an appropriate answer for a new question. "
        "Start your answer with \"Answer: \". Please answer the questions you are given directly without any "
        "introductory or concluding statements. "
        "Do not show reference and reference file when you generate the logic."
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=args.temperature,
        max_tokens=256,
    )

    response_logs = []
    model_response = response.choices[0].message.content.strip()
    response_logs.append(model_response)

    return response_logs




if __name__ == "__main__":
    print("Done.")