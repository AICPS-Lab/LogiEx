import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hscc-iccps26-paper254-repeatability-evaluations'))
import os
import pandas as pd
import numpy as np
from backend.algo.MCTS.utils import *
from backend.algo.MCTS.objects import *
from backend.algo.MCTS.requests import request_generator_real
from backend.algo.MCTS.routeplanner import *
import json


def seconds_to_hour(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"



def save_tree_info(tree, current_node=None, id=0):
    """Save data stored in MCTS search tree into a json file."""

    # Copied from the other file.
    # Previous requests: 
    # if status is dropped off, record actual drop off time and pick up time 
    # if status is in transit, record actual pick up time and est drop off
    # if status is assigned, record both est time for pick up and drop off

    ####################    
    # Get current node info.
    eta_stops = current_node.get_request_eta()
    # average_stops = current_node.get_average_stops()
    # str_stops = ""
    # for pair in average_stops:
    #     str_stops += str(pair[0]) + " stops before pick-up, " + str(pair[1]) + " stops before drop-off. "
    # print(str_stops)
    new_node = {
        "decision epoch": id, 
        "time": seconds_to_hour(current_node.t), 
        "current request": current_node.r_t.get_request_time(), 
        "assign to": current_node.get_assigned_vehicle(), 
        "eta": eta_stops[0], 
        "stops": str(eta_stops[1]) + " stops before pick-up; " + str(eta_stops[2]) + " stops before drop-off", 
        # "stops": str_stops,
        "previous requests": current_node.get_prev_requests(), 
        "vehicle status": current_node.get_vehicle_status(), 
        "N": tree.N[current_node], 
        "R": float(f"{tree.Q[current_node]:.2f}"), 
        "decomposed R": [float(f"{tree.breakdown_Q[current_node][0]:.2f}"), 
                         float(f"{tree.breakdown_Q[current_node][1]:.2f}")], 
        "violations": [],
        "children": []
    }

    if len(current_node.R) == 0:
        new_node["request"] = None
    try:
        children_nodes = tree.children[current_node]
        for j, curr_child in enumerate(children_nodes):
            if tree.N[curr_child] > 0:
                new_node["children"].append(save_tree_info(tree, curr_child, id+1))
        return new_node
    except KeyError:
        return new_node
    



def additional_search_data_files(cong: bool=False, request_arrive: int=60, window: int=10):
    FILE_FOLD = "backend/algo/data/"
    if cong:
        event_chains = pd.read_csv(os.path.join(FILE_FOLD,"CARTA/processed/train_chains_cong.csv"))
        travel_time_matrix = np.loadtxt(os.path.join(FILE_FOLD,"travel_time_matrix/travel_time_matrix_cong.csv"), delimiter=",")
        travel_time_matrix = travel_time_matrix.astype(int)
    else:
        event_chains = pd.read_csv(os.path.join(FILE_FOLD,"CARTA/processed/chains.csv"))
        travel_time_matrix = np.loadtxt(os.path.join(FILE_FOLD,"travel_time_matrix/travel_time_matrix.csv"), delimiter=",")
        travel_time_matrix = travel_time_matrix.astype(int)
    
    print("Loaded data for additional search.")
    event_chains = event_chains.sort_values(by='chain_id', ascending=True).drop_duplicates()
    event_chains['request_arrive_time'] = event_chains['pickup_time_since_midnight'].apply(lambda x: x - request_arrive*window)
    event_chains['e'] = event_chains['pickup_time_since_midnight']
    event_chains['l'] = event_chains['dropoff_time_since_midnight']
    
    return event_chains, travel_time_matrix




def additional_search_vehicle_routeplanner(
        num_vehicles: int=5, 
        capacity: int=3, 
        chain_id: int=9, 
        required_depth: int=10,
        current_time: int=10000,
        event_chains: pd.DataFrame=None, 
        travel_time_matrix: np.array=None,
        exclude_vehicle: int=-1):
    """
    Returns vehicle list and routeplanner. 
    Options are to disable some vehicles. 
    exclude_vehicle: -1 is not disabling vehicles.
    """

    v_list = []
    for i in range(num_vehicles):
        v_list.append(
            Vehicle(travel_time_matrix, num=i, seats=capacity)
        )

    for vehicle in v_list:
        if vehicle.id == exclude_vehicle:
            v_list.remove(vehicle)
    
    test_plan = RoutePlanner(
        chains=event_chains, travel_time_matrix=travel_time_matrix, 
        r_t=None, R=[], V=v_list, theta=dict(), t=current_time, 
        terminal=False, total_rq=required_depth, chain_id=chain_id
    )

    return v_list, test_plan



def do_additional_search(chain_id: int, request_id: int, vehicle_id: int, 
                         window: int=10, capacity: int=3, num_vehicles: int=15, 
                         exclude_vehicle: int=1, cong: bool=True, num_passengers: int=1,
                         random_existing_trips: bool=True):
    """
    Parameters: chain_id, request_id, vehicle_id
    request_id is the field `chain_order` in chain.csv
    assume vehicle needs additional search 
    assigning the current request to the id`ed vehicle 
    chain_id and request_id should be recorded somewhere. 
    num_vehicles and capacity should be the same as before. 
    random_existing_trips assigns existing trips to cars. 
    """
    
    required_depth = 10
    current_time = 10000
    print("Additional search parameters:", chain_id, request_id, vehicle_id,
          window, capacity, num_vehicles, exclude_vehicle, cong, num_passengers, required_depth)
    
    chains = {}
    event_chains, travel_time_matrix = additional_search_data_files(cong=cong, window=window)
    chains['train_chains'] = event_chains

    v_list, _ = additional_search_vehicle_routeplanner(
        num_vehicles=num_vehicles, 
        capacity=capacity, 
        chain_id=chain_id, 
        required_depth=required_depth, 
        current_time=current_time,
        event_chains=event_chains, 
        travel_time_matrix=travel_time_matrix,
        exclude_vehicle=exclude_vehicle
    )
    print("Created new route planner.")
    
    if random_existing_trips: 
        print("Generating random existing trips.")
        random_requests = [request_generator_real(event_chains, 0, 9)]
        current_request = None
        while not current_request: 
            current_request = check_requests(random_requests, current_time)
            current_time += 1
        print("Found a random request.")
        current_request.num_passengers = capacity
        route_plan, v_list = simple_assign(
            chains, travel_time_matrix, 
            current_time, current_request, 
            required_depth, num_vehicles, 
            capacity, saved_v_list=v_list)
        current_time = 10000
        request_id = 1
        requests = [request_generator_real(event_chains, 1, chain_id, relative_t=16000, spec_req_id=1)]
    else:
        current_time = 10000
        requests = [request_generator_real(event_chains, 0, chain_id)]

    current_request = None
    while not current_request: 
        current_request = check_requests(requests, current_time)
        current_time += 1
    current_request.num_passengers = num_passengers
    print("Found new request.")
    if random_existing_trips:
        print(route_plan.theta)
        print(str(route_plan.R[0]))
        tree, route_plan = assign_passenger(
            chains, travel_time_matrix, 
            current_time, current_request, 
            required_depth, num_vehicles, 
            capacity, saved_v_list=v_list, 
            defined_request=request_id, 
            defined_vehicle=vehicle_id,
            previous_rp=route_plan
        )
    else: 
        tree, route_plan = assign_passenger(
            chains, travel_time_matrix, 
            current_time, current_request, 
            required_depth, num_vehicles, 
            capacity, saved_v_list=v_list, 
            defined_request=request_id, 
            defined_vehicle=vehicle_id
        )
    
    search_tree_data = {}
    root_node = list(tree.children.keys())[0]
    search_tree_data = save_tree_info(tree, root_node, 0)
    
    os.makedirs('/tmp/transit_additional_search/', exist_ok=True)
    with open('/tmp/transit_additional_search/exp_test_{}.json'.format(request_id), 'w') as file:
        json.dump(search_tree_data, file, indent=4)

    # TODO: Change this to local directory if needed. 
    # with open('backend/data/transit_additional_search/exp_large_test_{}.json'.format(request_id), 'w') as file:
    #     json.dump(search_tree_data, file, indent=4)
    print(
        "Search tree file (additional search) saved to: "
        "/tmp/transit_additional_search/exp_test_{}.json".format(request_id)
    )
    if random_existing_trips:
        print(route_plan.theta)
        return "/tmp/transit_additional_search/exp_test_{}.json".format(request_id), route_plan.theta[1]

    return "/tmp/transit_additional_search/exp_test_{}.json".format(request_id), route_plan.theta[0]
    




if __name__ == "__main__":
    car_number = 0
    do_additional_search(9, 0, car_number)
    print("Done.")
    