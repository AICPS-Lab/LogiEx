"""
Reference: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

Implementation of route planner.
State: new request, assigned requests, vehicle locations, vehicle route plans.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hscc-iccps26-paper254-repeatability-evaluations'))
import re
import copy
import time
import random
import pandas as pd
from enum import Enum
from datetime import timedelta
from backend.algo.MCTS.mcts import MCTS, Node
from backend.algo.MCTS.objects import *
from backend.algo.MCTS.requests import request_generator_real




def add_minutes(initial_time, addition):
    hours, minutes = map(int, initial_time.split(':'))
    initial_time = timedelta(hours=hours, minutes=minutes)
    added_time = timedelta(minutes=addition)
    new_time = initial_time + added_time
    return new_time


status = Enum('status', 'waiting assigned in_transit dropped_off')
def seconds_to_hour(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"


def time_to_minutes_string(time_str):
    hours, minutes = map(int, time_str.split(':'))
    total_minutes = hours * 60 + minutes
    return f"{total_minutes} minutes"


class RoutePlanner(Node):
    """
    Formulate route planning as a MCTS problem.
    Defines state transitions.
    """

    def __init__(self, chains, travel_time_matrix, r_t, R, V, theta, t=0, 
                 terminal=False, total_rq=10, verbose=True, chain_id=1) -> None:
        super(Node, self).__init__()
        self.r_t = r_t      # new request
        self.R = R          # a list of existing requests 
        self.V = V          # a list of vehicles 
        self.theta = theta  # route plans
        self.t = t          # the current time of the state
        self.terminal = terminal
        self.total_rq = total_rq
        self.chains = chains
        self.travel_time_matrix = travel_time_matrix
        self.verbose = verbose
        self.chain_id = chain_id    # a random chain ID from train chains
        self.flag = False           # store if the conditions are violated
        self.violations = []        # store the reason for violation
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def is_terminal(self):
        return self.terminal
    
    def append_request(self, request):
        self.move_vehicle(request.request_time)
        return RoutePlanner(self.chains, self.travel_time_matrix, request, self.R, self.V, 
                            self.theta, request.request_time, self.terminal, self.total_rq, chain_id=self.chain_id)

    def find_children(self, check_violations=True):
        if self.terminal:   
            if check_violations:
                return set(), []
            else:
                return set()
        
        possible_plan = set()
        invalid_plan = []

        for v in self.V:
            curr_v = copy.deepcopy(v)
            curr_r = copy.deepcopy(self.r_t)
            if curr_v.assign_to_vehicle(curr_r, self.t, self.travel_time_matrix):
                next_state = self.make_move(curr_r, curr_v)
                possible_plan.add(next_state)
            else:
                invalid_plan.append((curr_r, curr_v, "vehicle_full"))
        
        if check_violations:
            return possible_plan, invalid_plan
        else:
            return possible_plan
    

    def get_children_status(self):
        possible_plan, invalid_plan = self.find_children()
        status_keys = {'current_status', 'pickup_time', 'dropoff_time', 'request_time',
                        'pickup_location', 'dropoff_location', 'actual_assigned_time', 
                        'actual_pickup_time', 'actual_dropoff_time'}
        
        children_ls = list(possible_plan)
        status_keys_ls = list(status_keys)
        children_status = dict()
        
        for key in status_keys:
            children_status[key] = []
        for c in children_ls:
            for key in status_keys_ls:
                children_status[key].append(c.R[-1].get_info(key))
        
        return children_status
    

    def children_spec(self, children):
        children_ls = list(children)
        for c in children_ls:
            self.check_child_spec(c)


    def find_random_child(self):
        if self.terminal:
            return None
        try:
            sample_child = random.sample([*self.find_children(check_violations=False)], 1)[0] 
            return sample_child
        except ValueError:
            raise RuntimeError("Request Rejected: increase vehicle capacity.")


    # def reward(self, break_down=False, a=1.0, b=1.0):
    #     trip_fullfill = 0
    #     timing = 0.0
    #     for i in range(len(self.R)):
    #         if self.R[i].current_status.name == "in_transit":
    #             trip_fullfill += 1
    #             timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000

    #         elif self.R[i].current_status.name == "dropped_off":
    #             trip_fullfill += 1
    #             timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000
    #             timing += (self.R[i].dropoff_time - self.R[i].actual_dropoff_time) / 10000

    #     if not break_down:
    #         return a*(trip_fullfill/len(self.R)) + b*timing
    #     else:
    #         return (a*(trip_fullfill/len(self.R)), b*timing)
        
    # the new reward.
    def reward(self, break_down=False, a=1.0, b=1.0):
        trip_fullfill = 0
        timing = 0.0
        for i in range(len(self.R)):
            if self.R[i].current_status.name == "in_transit":
                trip_fullfill += 1
                timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000
                if self.R[i].actual_pickup_time < self.R[i].pickup_time:
                    timing += (self.R[i].actual_pickup_time + 900 - self.R[i].pickup_time) / 10000
            
            elif self.R[i].current_status.name == "dropped_off":
                trip_fullfill += 1
                timing += (self.R[i].pickup_time - self.R[i].actual_pickup_time) / 10000
                timing += (self.R[i].dropoff_time - self.R[i].actual_dropoff_time) / 10000
                if self.R[i].actual_pickup_time < self.R[i].pickup_time:
                    timing += (self.R[i].actual_pickup_time + 900 - self.R[i].pickup_time) / 10000
                if self.R[i].actual_dropoff_time < self.R[i].dropoff_time:
                    timing += (self.R[i].actual_dropoff_time + 900 - self.R[i].dropoff_time) / 10000

        if not break_down:
            return a*(trip_fullfill/len(self.R)) + b*timing
        else:
            return (a*(trip_fullfill/len(self.R)), b*timing)


    def move_vehicle(self, next_t):
        for vh in self.V:
            vh.evolve_time(self.t, next_t, self.R, self.travel_time_matrix)


    def make_move(self, r, v):
        """
        Returns a new state of the board by assgining a request to a vehicle.
        r: a request that is currently being processed. 
        v: a vehicle object that has an updated route plan.
        [v = None] indicates the request cannot be assigned to any vehicle (rejected).
        """
        # next request
        next_rt = request_generator_real(self.chains, self.r_t.id+1, self.chain_id, self.t+1)
        if not next_rt.request_time > self.t:
            print(next_rt.request_time, self.t, next_rt)
        assert (next_rt.request_time > self.t)
        
        next_t = next_rt.request_time
        next_r = copy.deepcopy(self.R) 
        next_v = [copy.deepcopy(vi) for vi in self.V]
        
        for i in range(len(next_v)):
            if next_v[i].id == v.id:
                next_v[i] = v
        
        for vh in next_v:
            vh.evolve_time(self.t, next_t, next_r, self.travel_time_matrix)

        if len(self.R) > self.total_rq:   
            self.terminal = True
        
        next_r.append(r)
        new_theta = copy.deepcopy(self.theta)
        new_theta[r.id] = v.id
        return RoutePlanner(self.chains, self.travel_time_matrix, next_rt, next_r, next_v, 
                            new_theta, next_t, self.terminal, self.total_rq, chain_id=self.chain_id)
    

    def get_request_by_id(self, rid):
        for request in self.R:
            if request.id == rid:
                return request
        print("rid", rid)
        [print(request.id, end="; ") for request in self.R]
        return None


    def get_vehicle_status(self):
        return "".join(v.get_occupancy() for v in self.V)
    

    def get_average_stops(self):
        """
        Working code: Get average numbers of additional stops for all requests. 
        """
        if len(self.R) == 0: 
            return [[0, 0]]
        
        stops_for_every_r = []
        for r in self.R:
            recent = r
            assigned_to = None  # vehicle being assigned to
            for vehicle in self.V:
                if vehicle.id == recent.assigned_vehicle:
                    assigned_to = vehicle
            new_travel_plan = copy.deepcopy(assigned_to.travel_plan)

            # convert all requests to stop numbers
            pstop = recent.pickup_location
            dstop = recent.dropoff_location
            stop_numbers = []
            for i in range(len(new_travel_plan)):
                if new_travel_plan[i][0] == -1:
                    stop_numbers.append([assigned_to.current_location, -1])
                elif new_travel_plan[i][1] >= 1:
                    stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).pickup_location,new_travel_plan[i][0]])
                elif new_travel_plan[i][1] <= -1:
                    stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).dropoff_location,new_travel_plan[i][0]])
            
            pick_stops = 0
            drop_stops = 0
            if recent.current_status.name == "assigned":
                flag = True
                for i in range(len(stop_numbers)):
                    if stop_numbers[i][0] == pstop and stop_numbers[i][1] == recent.id:
                        drop_stops += 1
                        flag = False
                    elif stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                        stops_for_every_r.append([pick_stops, drop_stops])
                    else:
                        drop_stops += 1
                        if flag == True:
                            pick_stops += 1
                            
            elif recent.current_status.name == "in_transit":
                for i in range(len(stop_numbers)):
                    if stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                        stops_for_every_r.append([0, drop_stops])
                    else:
                        drop_stops += 1

            elif recent.current_status.name == "dropped_off": 
                stops_for_every_r.append([0,0])
            elif recent.current_status.name == "waiting": 
                stops_for_every_r.append([0,0])
            else:
                print("Undefined trip status.")

        return stops_for_every_r
    


    def get_request_eta(self):
        if len(self.R) == 0:
            return "Not Assigned", 0, 0

        pick_eta = 0
        drop_eta = 0
        recent = self.R[-1]
        assigned_to = None  # vehicle being assigned to
        for vehicle in self.V:
            if vehicle.id == recent.assigned_vehicle:
                assigned_to = vehicle
        new_travel_plan = copy.deepcopy(assigned_to.travel_plan)
        new_travel_plan.insert(0, (-1, -1))
        
        # convert all requests to stop numbers
        pstop = recent.pickup_location
        dstop = recent.dropoff_location
        stop_numbers = []
        for i in range(len(new_travel_plan)):
            if new_travel_plan[i][0] == -1:
                stop_numbers.append([assigned_to.current_location, -1])
            elif new_travel_plan[i][1] >= 1:
                stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).pickup_location,new_travel_plan[i][0]])
            elif new_travel_plan[i][1] <= -1:
                stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).dropoff_location,new_travel_plan[i][0]])
        
        pick_stops = 0
        drop_stops = 0
        if recent.current_status.name == "assigned":
            flag = True
            for i in range(len(stop_numbers)):
                if stop_numbers[i][0] == pstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        pick_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    drop_stops += 1
                    flag = False
                elif stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    return "pick-up in "+str(time_to_minutes_string(seconds_to_hour(pick_eta)))+"; drop-off in "+str(time_to_minutes_string(seconds_to_hour(drop_eta))), pick_stops, drop_stops
                else:
                    drop_stops += 1
                    if flag == True:
                        pick_stops += 1
                        if i != 0:
                            pick_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                        

        elif recent.current_status.name == "in_transit":
            for i in range(len(stop_numbers)):
                if stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    return "pick-up in "+str(time_to_minutes_string(seconds_to_hour(pick_eta)))+"; drop-off in "+str(time_to_minutes_string(seconds_to_hour(drop_eta))), 0, drop_stops
                else:
                    drop_stops += 1
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]

        elif recent.current_status.name == "dropped_off": 
            return "Fulfilled", 0, 0
        elif recent.current_status.name == "waiting": 
            return "Not Assigned", 0, 0
        return "Undefined", 0, 0
    

    def get_all_request_eta(self, rid):
        if len(self.R) == 0:
            return "Not Assigned", 0, 0
        
        pick_eta = 0
        drop_eta = 0
        recent = self.R[rid]
        assigned_to = None  # vehicle being assigned to
        for vehicle in self.V:
            if vehicle.id == recent.assigned_vehicle:
                assigned_to = vehicle
        new_travel_plan = copy.deepcopy(assigned_to.travel_plan)
        new_travel_plan.insert(0, (-1, -1))
        
        pstop = recent.pickup_location
        dstop = recent.dropoff_location
        stop_numbers = []
        for i in range(len(new_travel_plan)):
            if new_travel_plan[i][0] == -1:
                stop_numbers.append([assigned_to.current_location, -1])
            elif new_travel_plan[i][1] >= 1:
                stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).pickup_location,new_travel_plan[i][0]])
            elif new_travel_plan[i][1] <= -1:
                stop_numbers.append([self.get_request_by_id(new_travel_plan[i][0]).dropoff_location,new_travel_plan[i][0]])
        
        pick_stops = 0
        drop_stops = 0
        
        if recent.current_status.name == "assigned":
            flag = True
            for i in range(len(stop_numbers)):
                if stop_numbers[i][0] == pstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        pick_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    drop_stops += 1
                    flag = False
                elif stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    return "pick-up in "+str(time_to_minutes_string(seconds_to_hour(pick_eta)))+"; drop-off in "+str(time_to_minutes_string(seconds_to_hour(drop_eta))), pick_stops, drop_stops
                else:
                    drop_stops += 1
                    if flag == True:
                        pick_stops += 1
                        if i != 0:
                            pick_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
            print(new_travel_plan)
            print("assigned errors", stop_numbers, recent.id, dstop)
                        
        elif recent.current_status.name == "in_transit":
            for i in range(len(stop_numbers)):
                if stop_numbers[i][0] == dstop and stop_numbers[i][1] == recent.id:
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
                    return "pick-up in "+str(time_to_minutes_string(seconds_to_hour(pick_eta)))+"; drop-off in "+str(time_to_minutes_string(seconds_to_hour(drop_eta))), 0, drop_stops
                else:
                    drop_stops += 1
                    if i != 0:
                        drop_eta += self.travel_time_matrix[ stop_numbers[i-1][0],stop_numbers[i][0] ]
            print("in_transit errors", stop_numbers, recent.id, dstop)

        elif recent.current_status.name == "dropped_off": 
            return "Fulfilled", 0, 0
        elif recent.current_status.name == "waiting": 
            return "Not Assigned", 0, 0
        
        # this error message should not be displayed
        print("Trip status is undefined!!!", recent.current_status.name)
        return "Undefined", 0, 0
    

    def get_prev_requests(self):
        "Get status of ALL requests, including the current one."
        "If prev request is dropped off, return the actual time."
        "If prev request is in transit, return actual p time and eta."
        "If prev request is assigned, return eta."
        results = []

        if len(self.R) > 0:
            for rid, req in enumerate(self.R):
                result_string = ""
                
                if req.current_status.name == "dropped_off": 
                    etap = seconds_to_hour(req.actual_pickup_time)
                    etad = seconds_to_hour(req.actual_dropoff_time)

                elif req.current_status.name == "in_transit": 
                    etap = seconds_to_hour(req.actual_pickup_time)
                    
                    current_time = seconds_to_hour(self.t)
                    eta_string = self.get_all_request_eta(rid)[0]
                    pattern = r"drop-off in (\d+)"
                    match = re.search(pattern, eta_string)
                    if match:
                        etad = add_minutes(current_time, int(match.group(1)))
                    else:
                        print("req", req)
                        print("eta_string", eta_string)
                        raise ValueError("No valid 'drop-off' time found.")

                elif req.current_status.name == "assigned": 
                    current_time = seconds_to_hour(self.t)
                    
                    eta_string = self.get_all_request_eta(rid)[0]
                    pattern = r"pick-up in (\d+)"
                    match = re.search(pattern, eta_string)
                    if match:
                        etap = add_minutes(current_time, int(match.group(1)))
                    else:
                        print("req", req)
                        print("eta_string", eta_string)
                        raise ValueError("No valid 'pick-up' time found.")
                    
                    # match pattern for drop-off time
                    pattern = r"drop-off in (\d+)"
                    match = re.search(pattern, eta_string)
                    if match:
                        etad = add_minutes(current_time, int(match.group(1)))
                    else:
                        print("req", req)
                        print("eta_string", eta_string)
                        raise ValueError("No valid 'drop-off' time found.") 
                
                if isinstance(etap, timedelta):
                    result_string += "Pick-Up Time: " + seconds_to_hour(int(etap.total_seconds())) + ", "
                else:
                    result_string += "Pick-Up Time: " + str(etap) + ", "
                if isinstance(etad, timedelta):
                    result_string += "Drop-Off Time: " + seconds_to_hour(int(etad.total_seconds())) + ", "
                else:
                    result_string += "Drop-Off Time: " + str(etad) + ", "

                result_string += "Status: " + req.current_status.name
                results.append(result_string)
        
        else:
            return None
        
        ret = "; ".join(results)
        return ret
    


    def get_assigned_vehicle(self):
        if len(self.R) > 0:
            return "Car "+ str(self.R[-1].assigned_vehicle)
        else:
            return None

    
    def __str__(self):
        "Organize to print the current route plan. Information will be saved to JSON."
        result_string = "Current Time: {}. ".format(seconds_to_hour(self.t)) + "Assigning Passenger: {}. ".format(self.r_t) + "Previous Passengers: "
        if len(self.R) > 0:
            result_string += "; ".join(str(row) for row in self.R)
        else:
            result_string += "None. "
            
        result_string += "\nVehicle Fleet: \n" + "".join(str(v) for v in self.V)
        return result_string 



def simple_assign(chains, travel_time_matrix, current_time, r, num_requests, 
                  num_vehicles, capacity, saved_v_list=None):
    """
    Assign requests without the MCTS algorithm. 
    Returns a new route plan.
    r: current request to be assigned.
    """
    if not saved_v_list:
        vehicle_list = []
        for i in range(num_vehicles):
            vehicle_list.append(Vehicle(travel_time_matrix, num=i, seats=capacity))
    else:
        vehicle_list = saved_v_list

    rand_chain_id = 9
    route_plan = RoutePlanner(
        chains=chains['train_chains'], travel_time_matrix=travel_time_matrix, 
        r_t=None, R=[], V=vehicle_list, theta=dict(), t=current_time, terminal=False, 
        total_rq=num_requests, chain_id=rand_chain_id)
    route_plan = route_plan.append_request(r)
    next_plan = route_plan.find_random_child()
    return next_plan, next_plan.V



def assign_passenger(chains, travel_time_matrix, current_time, r, num_requests, 
                     num_vehicles, capacity, saved_v_list=None, defined_request=None, 
                     defined_vehicle=None, previous_R=None, previous_rp=None):
    """
    Do MCTS search for one request.
    Returns the expanded tree and a simulated route plan.
    defined_vehicle: force the passenger to be assigned to a vehicle.
    r: current request to be assigned.
    defined_request: must provide. 
    previous_R: existing requests. 
    """
    
    print("Initiating new MCTS.")
    tree = MCTS()

    if not saved_v_list:
        vehicle_list = [Vehicle(travel_time_matrix, num=i, seats=capacity) for i in range(num_vehicles)]
    else:
        vehicle_list = saved_v_list

    print("Vehicle initial location:")
    [print(v.current_location) for v in vehicle_list]
    
    rand_chain_id = random.randint(0,99)
    if previous_rp:
        route_plan = previous_rp
        r.id = 1
        defined_request = 1
    elif not previous_R:
        route_plan = RoutePlanner(
            chains=chains['train_chains'], 
            travel_time_matrix=travel_time_matrix, 
            r_t=None, R=[], V=vehicle_list, theta=dict(), 
            t=current_time, terminal=False, 
            total_rq=num_requests, chain_id=rand_chain_id
        )
    elif previous_R:
        route_plan = RoutePlanner(  
            chains=chains['train_chains'], 
            travel_time_matrix=travel_time_matrix, 
            r_t=None, R=previous_R, V=vehicle_list, theta=dict(), 
            t=current_time, terminal=False, 
            total_rq=num_requests, chain_id=rand_chain_id
        )
    else:
        raise ValueError("Double check function parameters.")
    
    route_plan = route_plan.append_request(r)

    for _ in range(50):
        tree.run_mcts(route_plan, defined_request, defined_vehicle)    
    print("Finished generating MCTS.")

    route_plan, all_children = tree.choose(route_plan, get_children=True)
    return tree, route_plan



if __name__ == "__main__":
    request_arrive = 60
    window = 10

    FILE_FOLD = "../data/"
    
    train_chains = pd.read_csv(os.path.join(FILE_FOLD, "CARTA/processed/train_chains.csv"))
    test_chains = pd.read_csv(os.path.join(FILE_FOLD,  "CARTA/processed/test_chains.csv"))

    train_chains = train_chains.sort_values(by='chain_id', ascending=True).drop_duplicates()
    test_chains = test_chains.sort_values(by='chain_id', ascending=True).drop_duplicates()
    
    # get earliest pickup and latest drop off time
    train_chains['request_arrive_time'] = train_chains['pickup_time_since_midnight'].apply(lambda x: x - request_arrive * 60)
    train_chains['e'] = train_chains['pickup_time_since_midnight'].apply(lambda x: x - window * 60)
    train_chains['l'] = train_chains['dropoff_time_since_midnight'].apply(lambda x: x + window * 60)
    
    test_chains['request_arrive_time'] = test_chains['pickup_time_since_midnight'].apply(lambda x: x - request_arrive * 60)
    test_chains['e'] = test_chains['pickup_time_since_midnight'].apply(lambda x: x - window * 60)
    test_chains['l'] = test_chains['dropoff_time_since_midnight'].apply(lambda x: x + window * 60)
    
    travel_time_matrix = np.loadtxt(os.path.join(FILE_FOLD,"travel_time_matrix/travel_time_matrix.csv"), delimiter=",")
    travel_time_matrix = travel_time_matrix.astype(int)
    nodes = pd.read_csv(os.path.join(FILE_FOLD,"travel_time_matrix/nodes.csv"))

    chains = {}
    chains['test_chains'] = test_chains
    chains['train_chains'] = train_chains

    # make route plan and record runtime
    begin_time = time.time()
    
    # build_plan(chains, travel_time_matrix)
    # print("elapsed time:", time.time()-begin_time)