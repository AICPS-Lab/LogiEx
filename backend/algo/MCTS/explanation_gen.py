import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hscc-iccps26-paper254-repeatability-evaluations'))
import os
import pandas as pd
import numpy as np
from backend.algo.MCTS.utils import check_requests
from backend.algo.MCTS.objects import *
from backend.algo.MCTS.requests import request_generator_real
from backend.algo.MCTS.tree_vis import graph_tree_vis
from backend.algo.MCTS.routeplanner import *
import json
import time


####################
# Hyperparameters: chain_id, num_cars, capacity, file starting index. 


def break_string(text, max_length=60):
    lines = []
    line = ""
    for word in text.split():
        if len(line) + len(word) <= max_length:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return lines



def seconds_to_hour(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"



def exploration_tree_query(tree, route_plan, request: int, vehicle: int, request_id: int):
    absolute_request = request  # the true request id represented in tree nodes
    text = "Q: Can you tell me more about assigning passenger {} to vehicle #{} instead?".format(absolute_request, vehicle)
    request = request-request_id  # the request count

    lines = break_string(text)
    for line in lines:
        print(line)
    
    prev_status = None
    max_visit_count = -1
    ####################
    # Identify the previous status, if exists.
    for k,v in tree.children.items():
        for vv in v:
            if len(vv.R) == request and tree.N[vv] > max_visit_count:
                max_visit_count = tree.N[vv]
                prev_status = vv

    visits = 0
    passenger_visits = []
    if not prev_status:
        for k, v in tree.children.items():
            for vv in v:
                if len(vv.R) == request + 1:
                    for assigned_rq in vv.R:
                        if assigned_rq.id == int(absolute_request):
                            passenger_visits.append(tree.N[vv])
                        if assigned_rq.assigned_vehicle == int(vehicle) and assigned_rq.id == int(absolute_request):
                            visits = tree.N[vv]
                            saved_routeplan = vv
    else:
        for vv in tree.children[prev_status]:
            for assigned_rq in vv.R:
                if assigned_rq.id == int(absolute_request):
                    passenger_visits.append(tree.N[vv])
                if assigned_rq.assigned_vehicle == int(vehicle) and assigned_rq.id == int(absolute_request):
                    visits = tree.N[vv]
                    saved_routeplan = vv

    underexplored = False
    if visits < max(passenger_visits):
        underexplored = True

    if not underexplored:
        text = "A: Certainly. "
        text += "The routing algorithm has delved into " + str(int(visits))
        text += " distinct future scenarios. In each of these scenarios, "
        text += "passenger {} was assigned to vehicle number {}.".format(absolute_request, vehicle)
        lines = break_string(text)
        for line in lines:
            print(line)
        print("\n", end="")
        return 
    
    ####################
    # Continue to explore from the requested scenario.
    test_chain_id = 8
    depth_counter = 0   
    required_depth = 1
    current_time = 0
    info_tree = None

    v_list = []
    for i in range(num_vehicles):
        v_list.append(Vehicle(travel_time_matrix, num=i, seats=capacity)) 

    additional_test_plan = RoutePlanner(
                chains=test_chains, travel_time_matrix=travel_time_matrix, 
                r_t=None, R=[], V=v_list, theta=dict(), t=current_time, 
                terminal=False, total_rq=num_requests, chain_id=test_chain_id)
    
    # Generate the next request.
    requests = [request_generator_real(test_chains, depth_counter, test_chain_id)]

    while depth_counter < required_depth: 
        r = check_requests(requests, current_time)
        if r:
            print("Depth", depth_counter)
            additional_test_plan = additional_test_plan.append_request(r)
            tree, test_plan = assign_passenger(chains, travel_time_matrix, 
                                                            current_time, r, required_depth, 
                                                            num_vehicles, capacity, 
                                                            defined_request=absolute_request, 
                                                            defined_vehicle=vehicle)
            info_tree = tree
            assigned_v_id = test_plan.theta[depth_counter]
            additional_test_plan.theta = {**additional_test_plan.theta, **test_plan.theta}
            assign_real_passenger = additional_test_plan.V[assigned_v_id].assign_to_vehicle(
                r,current_time,travel_time_matrix)
            assert assign_real_passenger == True
            depth_counter += 1
            requests.append(request_generator_real(test_chains, depth_counter, test_chain_id))   
            current_time = requests[-1].request_time
        else:
            current_time += 1

    def check_violations_assign(tree, request: int, status: str, time: str, vehicle: int): 
        total_scenarios = 0
        checked_violations = []
        for k, v in tree.children.items():
            for vv in v:
                total_scenarios += 1
                if len(vv.violations) > 0:
                    for vlt in vv.violations:
                        if int(vlt[0]) == int(request) and vlt[1] == status and vlt[2] == time:
                            checked_violations.append(float(vlt[3]))

        return total_scenarios, checked_violations
    
    text = "A: The initial query you requested wasn't the one the planning algorithm had explored the most thoroughly. "
    text += "However, in response to your request, we conducted further exploration with additional samples. "
    text += "The results suggest that if we proceed with the scenario you mentioned, "

    total_scenarios, violation_dropoff_late = check_violations_assign(
        info_tree, depth_counter, "dropoff", "late", vehicle)
    total_scenarios, violation_dropoff_early = check_violations_assign(
        info_tree, depth_counter, "dropoff", "early", vehicle)
    total_scenarios, violation_pickup_late = check_violations_assign(
        info_tree, depth_counter, "pickup", "late", vehicle)

    if (len(violation_dropoff_late)+len(violation_dropoff_early)+len(violation_pickup_late)) == 0:
        text += "in all " + str(int(total_scenarios))
        text += " additional cases the routing algorithm has explored, "
        text += "the passenger is expected to be transported by vehicle {} with no delays.".format(vehicle)
    
    else:
        text += "in all " + str(int(total_scenarios))
        text += " additional cases the routing algorithm has explored, "
        text += "the following violation(s) of trip requirements are likely to occur."

        if len(violation_dropoff_late) > 0:
            time_vio = np.array(violation_dropoff_late)
            text += " Late dropoff: avg: {:.2f}, ({:.2f}, {:.2f}). ".format(
                np.mean(time_vio), time_vio.min(), time_vio.max())
            text += "Violation rate: {:.2f}.".format(len(violation_dropoff_late)/int(total_scenarios))
       
        if len(violation_dropoff_early) > 0:
            time_vio = np.array(violation_dropoff_early)
            text += " Early dropoff: avg: {:.2f}, ({:.2f}, {:.2f}). ".format(
                np.mean(time_vio), time_vio.min(), time_vio.max())
            text += "Violation rate: {:.2f}.".format(len(violation_dropoff_early)/int(total_scenarios))
        
        if len(violation_pickup_late) > 0:
            time_vio = np.array(violation_pickup_late)
            text += " Late pickup: avg: {:.2f}, ({:.2f}, {:.2f}). ".format(
                np.mean(time_vio), time_vio.min(), time_vio.max())
            text += "Violation rate: {:.2f}.".format(len(violation_pickup_late)/int(total_scenarios))

    lines = break_string(text)
    for line in lines:
        print(line)

    print("\n", end="")



def assign_tree_query(tree, route_plan, request: int, status: str, time: str):
    # ID the assigned vehicle 
    assigned_v = route_plan.theta[request]  

    if status == "dropoff": 
        text = "Q: Is it expected that Passenger {} will be dropped off {}?".format(request, time)
    elif status == "pickup": 
        text = "Q: Is it expected that Passenger {} will be picked up {}?".format(request, time)
    
    lines = break_string(text)
    for line in lines:
        print(line)
    
    context_info = []
    total_scenarios = 0
    checked_violations = []
    request_info = None
    for k,v in tree.children.items():
        for vv in v:
            total_scenarios += 1
            if len(vv.violations) > 0:
                for vlt in vv.violations:
                    if int(vlt[0]) == int(request) and vlt[1] == status and vlt[2] == time and vlt[-1] == int(assigned_v):
                        checked_violations.append(float(vlt[3]))
                        context_info.append(vv)
    
    for k in tree.children.keys():
        if len(k.R) == 0:
            total_scenarios += 1 
            if len(k.violations) > 0:
                for vlt in k.violations:
                    if int(vlt[0]) == int(request) and vlt[1] == status and vlt[2] == time and vlt[-1] == int(assigned_v):
                        checked_violations.append(float(vlt[3]))
                        context_info.append(k)
            break

    for k, v in tree.children.items():
        for vv in v:
            for req in vv.R:
                if req.id == int(request):
                    request_info = req
                    break
    
    if time == "late":
        ex_words = "delay"
    elif time == "early":
        ex_words = "ahead of schedule"
    
    if len(checked_violations) == 0:
        text = "A: In all " + str(int(total_scenarios))
        text += " cases the routing algorithm has explored, "
        
        if status == "dropoff":
            text += "the passenger will not experience an {} drop-off.".format(time)
        elif status == "pickup":
            text += "the passenger will not experience an {} pick-up.".format(time)
        
        lines = break_string(text)
        for line in lines:
            print(line)

    else:
        if status == "dropoff":
            hours = int(request_info.dropoff_time // 3600)
            minutes = int((request_info.dropoff_time % 3600) // 60)
            text = "A: The passenger's requested drop-off time is set for {}:{}. ".format(hours, minutes) 
            text += "After examining "+ str(int(total_scenarios))
            text += " different routing scenarios, the routing algorithm indicates that, on average, the passenger may experience a {} in drop-off time. ".format(ex_words)
            text += "The {} is estimated to be around {:.2f} minutes, with a degree of variability ranging from as little as {:.2f} minutes to as much as {:.2f} minutes.".format(
                ex_words, np.mean(checked_violations), min(checked_violations), max(checked_violations))
            text += " Violation rate: {:.2f}%%.".format((len(checked_violations)/int(total_scenarios))*100)
            lines = break_string(text)
            for line in lines:
                print(line)

        elif status == "pickup":
            hours = int(request_info.pickup_time // 3600)
            minutes = (request_info.pickup_time % 3600) // 60
            text = "A: The passenger's requested pick-up time is set for {}:{}. ".format(hours, minutes) 
            text += "After examining "+ str(int(total_scenarios))
            text += " different routing scenarios, the routing algorithm indicates that, on average, the passenger may experience a {} in pick-up time. ".format(ex_words)
            text += "The {} is estimated to be around {:.2f} minutes, with a degree of variability ranging from as little as {:.2f} minutes to as much as {:.2f} minutes.".format(
                ex_words, np.mean(checked_violations), min(checked_violations), max(checked_violations))
            text += " Violation rate: {:.2f}%%.".format((len(checked_violations)/int(total_scenarios))*100)
            lines = break_string(text)
            for line in lines:
                print(line)
        
        ####################
        # Get information about how many passengers are ahead 
        all_queued = []
        for c in context_info: 
            for r in c.R:
                if r.id == request and r.assigned_vehicle == assigned_v:
                    route_ls = [ it[0] for it in c.V[assigned_v].route]
                    all_queued.append( route_ls.index(request) )
        
        print(all_queued)
        print(np.mean(all_queued))
    
    print("\n", end="")



def alternate_tree_query(tree, route_plan, request: int, vehicle: int):
    text = "Q: Why wasn't passenger {} assigned to vehicle #{}?".format(request, vehicle)
    lines = break_string(text)
    for line in lines:
        print(line)
    
    compare_v = None
    for assigned_rq in route_plan.R:
        if assigned_rq.assigned_vehicle == int(vehicle) and assigned_rq.id == int(request):
            text = "Passenger {} has already been assigned to vehicle #{}.".format(request, vehicle)
            text += " Please feel free to submit another query."
            lines = break_string(text)
            for line in lines:
                print(line)
            print()
            return
        elif assigned_rq.id == int(request):
            compare_v = assigned_rq.assigned_vehicle

    visits = []
    breakdown_rewards = []
    warning_vio = []
    detailed_vio = []
    for k in tree.children.keys():
        if len(k.R) == 0:
            for assigned_rq in k.R: 
                if assigned_rq.assigned_vehicle == int(vehicle) and assigned_rq.id == int(request):
                    for vlt in k.violations:
                        if int(vlt[0]) == int(request) and vlt[-1] == int(vehicle):
                            warning_vio.append("A violation of a time requirement is identified when attempting to assign passenger {} to vehicle {}. ".format(request, vehicle))
                            detailed_vio.append(vlt)
                    if int(tree.N[k]) > 0:
                        breakdown_rewards.append(tree.breakdown_Q[k])
                        visits.append(tree.N[k])
            break
            
    for k, v in tree.children.items():
        for vv in v:
            for assigned_rq in vv.R:
                if assigned_rq.assigned_vehicle == int(vehicle) and assigned_rq.id == int(request):
                    for vlt in vv.violations:
                        if int(vlt[0]) == int(request) and vlt[-1] == int(vehicle):
                            warning_vio.append("A violation of a time requirement is identified when attempting to assign passenger {} to vehicle {}. ".format(request, vehicle))
                            detailed_vio.append(vlt)
                    if int(tree.N[vv]) > 0:
                        breakdown_rewards.append(tree.breakdown_Q[vv])
                        visits.append(tree.N[vv])
                          
    if len(warning_vio) > 0:
        text = warning_vio[0]
        ####################
        # Late Pick-Up
        late_pickup_time = []
        for i in range(len(detailed_vio)):
            if detailed_vio[i][1] == 'pickup' and detailed_vio[i][2] == 'late':
                late_pickup_time.append(detailed_vio[i][3])
        if len(late_pickup_time) > 0:
            late_pickup_time = np.array(late_pickup_time)
            mean_late_pickup = np.mean(late_pickup_time)
            text += "If we proceed with this assignment, "
            text += "passenger {} is estimated to be picked up late by approximately {:.2f} minutes, ".format(request, mean_late_pickup)
            text += "with a degree of variability ranging from as little as {:.2f} minutes to as much as {:.2f} minutes.".format(late_pickup_time.min(), late_pickup_time.max())
            text += " Violation rate: {:.2f}%%.".format((len(late_pickup_time)/int(sum(visits)))*100)

        #################### 
        # Early Drop-Off
        early_dropoff_time = []
        for i in range(len(detailed_vio)):
            if detailed_vio[i][1] == 'dropoff' and detailed_vio[i][2] == 'early':
                early_dropoff_time.append(detailed_vio[i][3])
        if len(early_dropoff_time) > 0:
            early_dropoff_time = np.array(early_dropoff_time)
            mean_early_dropoff = np.mean(early_dropoff_time)
            text += "If we proceed with this assignment, "
            text += "passenger {} is estimated to be dropped off early by approximately {:.2f} minutes, ".format(request, mean_early_dropoff)
            text += "with a degree of variability ranging from as little as {:.2f} minutes to as much as {:.2f} minutes.".format(early_dropoff_time.min(), early_dropoff_time.max())
            text += " Violation rate: {:.2f}%%.".format((len(early_dropoff_time)/int(sum(visits)))*100)
        
        #################### 
        # Late Drop-Off
        late_dropoff_time = []
        for i in range(len(detailed_vio)):
            if detailed_vio[i][1] == 'dropoff' and detailed_vio[i][2] == 'late':
                late_dropoff_time.append(detailed_vio[i][3])
        if len(late_dropoff_time) > 0:
            late_dropoff_time = np.array(late_dropoff_time)
            mean_late_dropoff = np.mean(late_dropoff_time)
            text += "If we proceed with this assignment, "
            text += "passenger {} is estimated to be dropped off late by approximately {:.2f} minutes, ".format(request, mean_late_dropoff)
            text += "with a degree of variability ranging from as little as {:.2f} minutes to as much as {:.2f} minutes.".format(late_dropoff_time.min(), late_dropoff_time.max())
            text += " Violation rate: {:.2f}%%.".format((len(late_dropoff_time)/int(sum(visits)))*100)

        lines = break_string(text)
        for line in lines:
            print(line)

    elif len(visits) > 0:
        text = "After a thorough analysis of the algorithm exploration history, "
        text += "we have identified the following reasons for not recommending the route plan you proposed. "

        assigned_visits = []
        assigned_breakdown_rewards = []
        for k, v in tree.children.items():
            for vv in v:
                for assigned_rq in vv.R:
                    if assigned_rq.assigned_vehicle == int(compare_v) and assigned_rq.id == int(request):
                        if int(tree.N[vv]) > 0:
                            assigned_visits.append(tree.N[vv])
                            assigned_breakdown_rewards.append(tree.breakdown_Q[vv])

        fl1 = [item for sublist in breakdown_rewards for item in sublist]
        fl2 = [item for sublist in assigned_breakdown_rewards for item in sublist]
        text += "When we compare it with the recommended route plan, which has a composite score of {:.2f}, ".format(sum(fl2))
        text += "your proposed plan scores {:.2f}. ".format(sum(fl1))
        text += "This lower score indicates that your plan performs worse and is not as favorable. " 

        t0_fl1 = [sublist[0] for sublist in breakdown_rewards]
        t0_fl2 = [sublist[0] for sublist in assigned_breakdown_rewards]
        text += "Specifically, the recommended plan shows a {:.2f}%% improvement in service rate".format((sum(t0_fl2)-sum(t0_fl1))/sum(t0_fl1)*100)
        text += ", allowing it to accommodate more trips efficiently."

        t1_fl1 = [sublist[1] for sublist in breakdown_rewards]
        t1_fl2 = [sublist[1] for sublist in assigned_breakdown_rewards]
        text += " The recommended plan also exhibits a {:.2f}%% improvement in adhering to the required trip time".format((sum(t1_fl2)-sum(t1_fl1))/sum(t1_fl1)*100)
        text += ", ensuring that more passengers are transported punctually." 
        
        lines = break_string(text)
        for line in lines:
            print(line)
    print("\n", end="")



def generate_user_query(tree, route_plan, request_id):
    # Define user queries. 
    new_user_query = [
        "assign_r0_dropoff_late", 
        "assign_r0_dropoff_early", 
        "assign_r0_pickup_late", 
        "alternate_r0_v0", 
        "alternate_r0_v1", 
        "alternate_r0_v2", 
        "alternate_r0_v3", 
        "alternate_r0_v4", 
        "alternate_r0_v5", 
        "alternate_r0_v6", 
        "alternate_r0_v7", 
        "explore_r0_v0", 
        "explore_r0_v1", 
        "explore_r0_v2", 
        "explore_r0_v3", 
        "explore_r0_v4", 
        "explore_r0_v5", 
        "explore_r0_v6", 
        "explore_r0_v7", 
    ]
    
    # The above code is a comment in Python. Comments are used to provide explanations or notes within
    # the code for better understanding. In this case, the comment is indicating that the code is
    # written in Python.
    runtimes = []
    for q in new_user_query:
        start_t = time.time()
        q = q.split("_")
        if q[0] == "assign":
            assert (len(q) == 4),"Incomplete query!"
            q = q[1:]
            q_result = assign_tree_query(tree, route_plan, int(request_id), str(q[1]), str(q[2]))  
        elif q[0] == "alternate":
            q = q[1:]
            q_result = alternate_tree_query(tree, route_plan, int(request_id), int(q[1][1:]))
        elif q[0] == "explore":
            assert (len(q) == 3),"Incomplete query!" 
            q = q[1:]
            q_result = exploration_tree_query(tree, route_plan, int(request_id)+int(q[0][1:]), int(q[1][1:]), int(request_id))

        print(q[0], "time used:", time.time()-start_t)
        runtimes.append(time.time()-start_t)

    print(sum(runtimes)/len(new_user_query))
    return



def save_tree_info(tree, current_node=None, id=0):
    """Save data stored in MCTS search tree into a json file."""

    # Previous requests: 
    # if status is dropped off, record actual drop off time and pick up time 
    # if status is in transit, record actual pick up time and est drop off
    # if status is assigned, record both est time for pick up and drop off

    ####################
    # Get current node info.
    eta_stops = current_node.get_request_eta()
    new_node = {
        "decision epoch": id, 
        "time": seconds_to_hour(current_node.t), 
        "current request": current_node.r_t.get_request_time(), 
        "assign to": current_node.get_assigned_vehicle(), 
        "eta": eta_stops[0], 
        "stops": str(eta_stops[1]) + " stops before pick-up; " + str(eta_stops[2]) + " stops before drop-off", 
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



def plan_route(chains, travel_time_matrix, num_requests, num_vehicles, capacity, vehicle_full=None):
    "Generate the planned route and query explanations."
    request_counter = 0
    current_time = 10000
    test_chain_id = 16
    test_chains = chains['test_chains']
    test_chains = test_chains[test_chains['chain_id'] == test_chain_id]
    # Init a random request.
    requests = [request_generator_real(test_chains, request_counter, test_chain_id)]

    ####################
    # Create a vehicle list and a route plan.
    test_vehicle_list = []
    for i in range(num_vehicles):
        test_vehicle_list.append(Vehicle(travel_time_matrix, num=i, seats=capacity)) 

    if vehicle_full: 
        for vid in vehicle_full:
            test_vehicle_list[vid].occupancy = test_vehicle_list[vid].capacity

    ####################
    # Create a route plan.
    test_route_plan = RoutePlanner(
        chains=test_chains, travel_time_matrix=travel_time_matrix, 
        r_t=None, R=[], V=test_vehicle_list, theta=dict(), t=current_time, 
        terminal=False, total_rq=num_requests, chain_id=test_chain_id
    )

    ####################
    # Iterate through events.
    while request_counter < num_requests:
        r = check_requests(requests, current_time)
        
        if r:
            print("New Request:", r.id, r)
            test_route_plan = test_route_plan.append_request(r)
            tree, route_plan = assign_passenger(chains, travel_time_matrix, r.pickup_time-1500, 
                                                             r, num_requests, num_vehicles, capacity,
                                                             saved_v_list=test_route_plan.V)

            search_tree_data = {}
            root_node = list(tree.children.keys())[0]
            search_tree_data = save_tree_info(tree, root_node, 0)
            
            with open('backend/data/transit_additional_search/exp_eval_{}.json'.format(request_counter), 'w') as file:
                json.dump(search_tree_data, file, indent=4)
            print("Search Tree saved to: backend/data/transit_eval_json/exp_eval_{}.json".format(request_counter))
            
            # # generate and save tree visualization       
            # graph_tree_vis(tree, os.path.join(
            #     os.path.abspath(os.getcwd()), 
            #     "backend/data/transit_test_json",
            #     "search_tree_rq_{}.html".format(request_counter)))

            # # get vehicle information
            # if request_counter == 1:
                # get user query at this round
            # generate_user_query(tree, route_plan, request_counter)
            # [print(v) for v in route_plan.V]
            
            # move the real route plan to the next stage.
            assigned_v_id = copy.deepcopy(route_plan.theta[request_counter])
            print("MCTS Decision: Request", request_counter, "assigned to Car", assigned_v_id)
            assign_real_passenger = test_route_plan.V[
                assigned_v_id].assign_to_vehicle(r,current_time,travel_time_matrix)
            assert assign_real_passenger == True
            test_route_plan.theta = {**route_plan.theta, **test_route_plan.theta}

            request_counter += 1
            requests.append(request_generator_real(test_chains, request_counter, test_chain_id, current_time))   
            current_time = requests[-1].request_time
            
            if test_route_plan.is_terminal():
                break
        else:
            current_time += 1



if __name__ == "__main__":
    request_arrive = 60
    window = 10
    
    FILE_FOLD = "backend/algo/data/"
    
    ####################
    # load chains
    train_chains = pd.read_csv(os.path.join(FILE_FOLD,"CARTA/processed/chains.csv"))
    test_chains = pd.read_csv(os.path.join(FILE_FOLD,"CARTA/processed/chains.csv"))

    # preprocess chains
    train_chains = train_chains.sort_values(by='chain_id', ascending=True).drop_duplicates()
    test_chains = test_chains.sort_values(by='chain_id', ascending=True).drop_duplicates()
    
    # get earliest pickup and latest drop off time
    train_chains['request_arrive_time'] = train_chains['pickup_time_since_midnight'].apply(lambda x: x - request_arrive * 20)
    train_chains['e'] = train_chains['pickup_time_since_midnight']
    train_chains['l'] = train_chains['dropoff_time_since_midnight']
    
    test_chains['request_arrive_time'] = test_chains['pickup_time_since_midnight'].apply(lambda x: x - request_arrive * 20)
    test_chains['e'] = test_chains['pickup_time_since_midnight']
    test_chains['l'] = test_chains['dropoff_time_since_midnight']
    
    # load travel time matrix
    travel_time_matrix = np.loadtxt(os.path.join(FILE_FOLD,"travel_time_matrix/travel_time_matrix.csv"), delimiter=",")
    travel_time_matrix = travel_time_matrix.astype(int)

    chains = {}
    chains['test_chains'] = test_chains
    chains['train_chains'] = train_chains
    
    # load nodes
    nodes = pd.read_csv(os.path.join(FILE_FOLD,"travel_time_matrix/nodes.csv"))

    num_requests = 10
    num_vehicles = 15
    capacity = 2
    plan_route(
        chains, 
        travel_time_matrix, 
        num_requests, 
        num_vehicles, 
        capacity
    )
