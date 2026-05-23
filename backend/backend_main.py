import json
import os
import argparse as argument_parser
import backend.src.transit.openai_calls as transit_main 
import backend.prompts.test_prompting_transit as transit_logic_main 
import backend.tasks.transit as transit_logic_task
import backend.utils as utils





def open_file_by_scenario(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("Data file:", file_path)
            return data
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")




def run(args, query, thread_id, scenario_number):
    print("Thread ID (main):", thread_id)
    print("Scenario number:", scenario_number)

    if scenario_number == 1:
        file_path = 'backend/data/transit_test_json/exp_final_0.json'
    elif scenario_number == 2:
        file_path = 'backend/data/transit_test_json/exp_final_1.json'
    elif scenario_number == 3:
        file_path = 'backend/data/transit_test_json/exp_final_2.json'   
    data = open_file_by_scenario(file_path)

    gpt_client, gpt_assistant, thread = transit_main.llm_init_wo_thread(args.backend, args.api_key, None, thread_id)
    
    if args.logic_enabled == True:
        # DO NOT DELETE
        # vehicle_numbers = transit_logic_main.extract_vehicle_numbers(query)
        # query_type = transit_logic_main.match_query(query, vehicle_numbers)
        query_type = transit_logic_main.llm_match_query(query, gpt_client, gpt_assistant, args)
        print(f"Query: {query} - Type: {query_type}")
        
        if query_type == -1:
            final_response = transit_logic_task.simple_rag_qa(query, gpt_client, gpt_assistant, args)
        
        else:
            prompt = transit_logic_main.load_prompt(query_type, query)
            gpt_reply = transit_logic_main.test_one_logic_prompt(args, gpt_client, gpt_assistant, prompt)
            logic_checking_results = transit_logic_task.process_llm_answer(gpt_reply, data, scenario_number-1)

            if logic_checking_results == "ParseError":
                print("Parse Error returned.")
                final_response = transit_logic_task.simple_rag_qa(query, gpt_client, gpt_assistant, args)
            
            else:
                gpt_reply = gpt_reply[0][8:]
                print("Logic:", gpt_reply)
                gpt_reply = "Logic: " + gpt_reply + "\n"
                original_query = "Query: " + query + "\n"
                logic_checking_results = "Logic Checking Results: " + str(logic_checking_results)
                final_response = transit_logic_task.generate_final_answer(
                    args, gpt_client, gpt_assistant, 
                    gpt_reply+original_query+logic_checking_results)
            
        return final_response[0], thread.id
        
    else:
        queries = transit_main.load_queries(args.query_start_index, args.query_end_index)
        for i in range(len(queries)): 
            print("\nQuery index:", i+args.query_start_index)
            vector_store_id, upload_file_id = transit_main.load_tree_file_openai(gpt_client, (i+args.query_start_index)//5)
            gpt_reply = transit_main.test_one_query(args, gpt_client, gpt_assistant, queries[i])
            utils.delete_file(gpt_client, vector_store_id, upload_file_id)
            print("\n")





def parse_known_args(defaults=None):
    args = argument_parser.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'gpt-4', 'gpt-3.5-turbo'], default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--api_key', type=str, default=os.getenv('OPENAI_API_KEY'))
    args.add_argument('--task', type=str, choices=['transit'], default='transit')
    args.add_argument('--logic_enabled', action='store_true', help='disable logic processing.', default=True)
    args.add_argument('--query_start_index', type=int, default=19)
    args.add_argument('--query_end_index', type=int, default=20)
    args.add_argument('--repeat', type=int, default=1)
    args.add_argument('--prompt_style', type=str, choices=['standard', 'cot', 'ltt'], default='standard')
    args, _ = args.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_known_args()
    print(args)
    run(args)