import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json
from backend.tasks.transit_logics import *
from backend.algo.MCTS.additional_search import *

if (sys.version_info > (3, 0)):
    from functools import singledispatch
else:
    from singledispatch import singledispatch
import re
from datetime import datetime, timedelta
import time 




def average_time_difference(time_str_list, time_str2, type='d', time_window=0):
    format_str = '%H:%M'
    today = datetime.today()
    total_time_delta = timedelta()
    count = 0

    for i in range(len(time_str_list)):
        time_str1 = time_str_list[i]
        datetime_obj1 = datetime.strptime(time_str1, format_str).time()
        datetime_obj1 = datetime.combine(today, datetime_obj1)
        datetime_obj2 = datetime.strptime(time_str2, format_str).time()
        datetime_obj2 = datetime.combine(today, datetime_obj2)
        if type == 'd':
            time_delta = datetime_obj1 - (timedelta(minutes=time_window) + datetime_obj2)
        elif type == 'a':
            time_delta = datetime_obj2 - (datetime_obj1 + timedelta(minutes=time_window))

        if time_delta.total_seconds() > 0:
            total_time_delta += time_delta
            count += 1

    if count == 0:
        return "00:00"

    average_time_delta = total_time_delta / count
    total_seconds = int(average_time_delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    formatted_average_time = f"{hours:02}:{minutes:02}"

    return formatted_average_time




def time_difference(time_str1, time_str2, window=0):
    format_str = '%H:%M'
    datetime_obj1 = datetime.strptime(time_str1, format_str)
    datetime_obj2 = datetime.strptime(time_str2, format_str)
    time_delta = datetime_obj1 - datetime_obj2

    if time_delta.total_seconds() < 0:
        formatted_time = "00:00"
    else:
        total_seconds = int(time_delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        formatted_time = f"{hours:02}:{minutes:02}"
    return formatted_time



def calculate_true_percentage(values):
    true_count = sum(1 for element in values if element is True)
    total_elements = len(values)
    if total_elements == 0:
        return 0 
    true_percentage = (true_count / total_elements) * 100
    return true_percentage



def bool_delay_status(node, check_type, check_vehicle, req_time, time_window=0):
    all_requests = [item.strip() for item in node["previous requests"].split(';')]
    current_request = [item.strip() for item in all_requests[0].split(',')]
    
    if isinstance(check_type, PickUpTime):
        text = current_request[0]
        pattern = r"Pick-Up Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid pick-up time found.")
    
    elif isinstance(check_type, DropOffTime):
        text = current_request[1]
        pattern = r"Drop-Off Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid drop-off time found.")
    
    today = datetime.today()
    time_ref = datetime.strptime(req_time, "%H:%M").time() 
    time_ref = datetime.combine(today, time_ref) + timedelta(minutes=time_window)
    time_real = datetime.strptime(real_fulfill_time, "%H:%M").time()
    time_real = datetime.combine(today, time_real)

    if time_ref < time_real:
        return True
    else:
        return False



def bool_adv_status(node, check_type, check_vehicle, req_time, time_window=0):
    all_requests = [item.strip() for item in node["previous requests"].split(';')]
    current_request = [item.strip() for item in all_requests[0].split(',')]
    
    if isinstance(check_type, PickUpTime):
        text = current_request[0]
        pattern = r"Pick-Up Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid pick-up time found.")
    
    elif isinstance(check_type, DropOffTime):
        text = current_request[1]
        pattern = r"Drop-Off Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid drop-off time found.")

    today = datetime.today()
    time_ref = datetime.strptime(req_time, "%H:%M").time()
    time_ref = datetime.combine(today, time_ref)
    time_real = datetime.strptime(real_fulfill_time, "%H:%M").time() 
    time_real = datetime.combine(today, time_real) + timedelta(minutes=time_window)

    if time_ref > time_real:
        return True
    else:
        return False



def check_delay(node, check_type, check_vehicle, reference_time, time_window):
    # this function checks the real fulfillment time, not just the delay
    all_requests = [item.strip() for item in node["previous requests"].split(';')]
    current_request = [item.strip() for item in all_requests[0].split(',')]
    
    if isinstance(check_type, PickUpTime):
        text = current_request[0]
        pattern = r"Pick-Up Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid pick-up time found.")

    elif isinstance(check_type, DropOffTime):
        text = current_request[1]
        pattern = r"Drop-Off Time:\s*(\d{2}:\d{2})"
        match = re.search(pattern, text)
        if match:
            real_fulfill_time = match.group(1)
        else:
            print("No valid drop-off time found.")
    return real_fulfill_time



def enumerate_nodes(node, 
                    results, 
                    checking_function=check_delay, 
                    check_type=None, 
                    check_vehicle=0,
                    reference_time=None,
                    time_window=0):
    if node["decision epoch"] > 0:
        pattern = r"Car (\d+)"
        text = node["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if car_number == check_vehicle:
                result = checking_function(
                    node, 
                    check_type, 
                    check_vehicle, 
                    reference_time,
                    time_window
                )
                results.append(result)

    if len(node["children"]) > 0:
        for child in node["children"]:
            child_results = enumerate_nodes(
                child, 
                results, 
                checking_function, 
                check_type, 
                check_vehicle,
                reference_time,
                time_window
            )
            results = child_results

    return results



def time_to_minutes(t):
    return datetime.strptime(t, '%H:%M').hour * 60 + datetime.strptime(t, '%H:%M').minute



@singledispatch
def quantitativescore(transit_logics, data, scenario_num):
    raise TypeError("No quantitativescore for {} of class {}".format(transit_logics, transit_logics.__class__))



@quantitativescore.register(AvailableCar)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    res = 0
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            rule = 'vcv(C({}), O(1,{}))'.format(car_number,car_number)
            parsed = parse(rule)
            if qualitativescore(parsed, data, scenario_num) == False: 
                res += 1
    return res



@quantitativescore.register(CarAssign)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    assigned_car_num = 0
    max_r = -1000
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if float(child["R"]) > max_r: 
                max_r = float(child["R"])
                assigned_car_num = car_number
    return assigned_car_num



@quantitativescore.register(PickUpTime)
def _(transit_logics, data, scenario_num):
    text = data["current request"]
    pattern = r"Pick-Up Time: (\d{2}:\d{2})"
    match = re.search(pattern, text)
    if match:
        pick_up_time = match.group(1)
        return pick_up_time
    else:
        pass
        # print("Cannot find pick-up time.")
    


@quantitativescore.register(DropOffTime)
def _(transit_logics, data, scenario_num):
    text = data["current request"]
    pattern = r"Drop-Off Time: (\d{2}:\d{2})"
    match = re.search(pattern, text)
    if match:
        drop_off_time = match.group(1)
        return drop_off_time
    else:
        pass
        # print("Cannot find drop-off time.")
    


@quantitativescore.register(TreeVisit)
def _(transit_logics, data, scenario_num):
    if transit_logics.dep == 0:
        return int(data["N"])
    current_tree = data
    while True:
        if int(current_tree["decision epoch"]) == transit_logics.dep:
            return int(current_tree["N"]) 
        elif int(current_tree["decision epoch"]) < transit_logics.dep:
            for c in current_tree["children"]:
                if int(c["assign to"][4:]) == transit_logics.vehicle:
                    current_tree = c
        else:
            pass
            # print("Cannot find anything with the specified tree depth.")
    


@quantitativescore.register(Capacity)
def _(transit_logics, data, scenario_num):
    text = data["vehicle status"]
    car_number = transit_logics.vehicle
    pattern = rf"Car {car_number}:.*?capacity=(\d+)"
    match = re.search(pattern, text)
    if match:
        capacity = match.group(1)
        return int(capacity)
    else:
        pass
        # print("Cannot find capacity.")



@quantitativescore.register(Congestion)
def _(transit_logics, data, scenario_num):
    # print("Performing additional search under congestion.")
    vehicle_fleet_size = [5,4,8]
    capacities = [3,4,2]
    file_path,car_number = do_additional_search(
        9, 0, 0, 
        capacity=capacities[scenario_num],
        num_vehicles=vehicle_fleet_size[scenario_num],
        cong=True
    )
    # print("Additional search file saved to:", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        pass
        # print("The file was not found.")
    except json.JSONDecodeError:
        pass
        # print("Failed to decode JSON.")
    # Human-defined rules for additional search.
    rules = [
        'vioa(tp({}),eta({}))'.format(car_number,car_number),
        'viod(tp({}),eta({}))'.format(car_number,car_number),
        'vioa(td({}),eta({}))'.format(car_number,car_number), 
        'viod(td({}),eta({}))'.format(car_number,car_number),
    ]
    result = []
    for logic in rules: 
        parsed = parse(logic)
        result.append(logic)
        result.append(quantitativescore(parsed, data, scenario_num))
    return result



@quantitativescore.register(Exclude)
def _(transit_logics, data, scenario_num):
    print("Performing additional search under congestion.")
    exc_vehicle = transit_logics.vehicle 
    vehicle_fleet_size = [5,4,8]
    capacities = [3,4,2]
    file_path,car_number = do_additional_search(
        9, 0, 0, 
        capacity=capacities[scenario_num], 
        cong=False, 
        num_vehicles=vehicle_fleet_size[scenario_num],
        exclude_vehicle=exc_vehicle
    )
    print("Additional search file saved to:", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")
    
    rules = [
        'vioa(tp({}),eta({}))'.format(car_number,car_number),
        'viod(tp({}),eta({}))'.format(car_number,car_number),
        'vioa(td({}),eta({}))'.format(car_number,car_number), 
        'viod(td({}),eta({}))'.format(car_number,car_number),
    ]
    result = []
    result.append("Trip assigned to vehicle {}.".format(car_number))
    for logic in rules: 
        print(logic)
        parsed = parse(logic)
        print(quantitativescore(parsed, data, scenario_num))
        result.append(logic)
        result.append(quantitativescore(parsed, data, scenario_num))
    return result



@quantitativescore.register(Reassign)
def _(transit_logics, data, scenario_num):
    car_number = transit_logics.vehicle
    vehicle_fleet_size = [5,4,8]
    capacities = [3,4,2]
    file_path, _ = do_additional_search(
        9, 0, car_number, capacity=capacities[scenario_num],
        num_vehicles=vehicle_fleet_size[scenario_num],
        cong=False
    )
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")
    rules = [
        'vioa(tp({}),eta({}))'.format(car_number,car_number),
        'viod(tp({}),eta({}))'.format(car_number,car_number),
        'vioa(td({}),eta({}))'.format(car_number,car_number),
        'viod(td({}),eta({}))'.format(car_number,car_number),
    ]
    result = []
    result.append("Trip reassigned to vehicle {}.".format(car_number))
    for logic in rules:
        parsed = parse(logic)
        result.append(logic)
        result.append(quantitativescore(parsed, data, scenario_num))
    return result



@quantitativescore.register(MultiPass)
def _(transit_logics, data, scenario_num):
    print("Performing additional search for multiple passengers in the same trip.")
    num_passengers = transit_logics.request 
    vehicle_fleet_size = [5,4,8]
    capacities = [3,4,2]
    try: 
        file_path,car_number = do_additional_search(
            9, 0, 0, cong=True, capacity=capacities[scenario_num],
            num_vehicles=vehicle_fleet_size[scenario_num],
            num_passengers=num_passengers
        )
        print("Additional search file saved to:", file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("The file was not found.")
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
        rules = [
            'vioa(tp({}),eta({}))'.format(car_number,car_number),
            'viod(tp({}),eta({}))'.format(car_number,car_number),
            'vioa(td({}),eta({}))'.format(car_number,car_number), 
            'viod(td({}),eta({}))'.format(car_number,car_number),
        ]
        result = []
        result.append("Trip assigned to vehicle {}.".format(car_number))
        for logic in rules: 
            print(logic)
            parsed = parse(logic)
            print(quantitativescore(parsed, data, scenario_num))
            result.append(logic)
            result.append(quantitativescore(parsed, data, scenario_num))
        return result
    except: 
        return "There is something wrong with the given condition."



@quantitativescore.register(AdditionalSearch)
def _(transit_logics, data, scenario_num):
    car_number = transit_logics.vehicle
    print("Performing additional search on vehicle", car_number)
    vehicle_fleet_size = [5,4,8]
    capacities = [3,4,2]
    file_path, _ = do_additional_search(
        9, 0, car_number, capacity=capacities[scenario_num], 
        num_vehicles=vehicle_fleet_size[scenario_num]
    )
    print("Additional search file saved to:", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")
    # Human-defined rules for additional search.
    rules = [
        'vioa(tp({}),eta({}))'.format(car_number,car_number),
        'viod(tp({}),eta({}))'.format(car_number,car_number),
        'vioa(td({}),eta({}))'.format(car_number,car_number), 
        'viod(td({}),eta({}))'.format(car_number,car_number),
    ]
    result = []
    for logic in rules: 
        print(logic)
        parsed = parse(logic)
        print(quantitativescore(parsed, data, scenario_num))
        result.append(logic)
        result.append(quantitativescore(parsed, data, scenario_num))
    return result




@quantitativescore.register(Occupancy)
def _(transit_logics, data, scenario_num):
    text = data["vehicle status"]
    car_number = transit_logics.vehicle
    pattern = rf"Car {car_number}:.*?occupancy=(\d+)"
    match = re.search(pattern, text)
    if match:
        occupancy = match.group(1)
        return int(occupancy)
    else:
        pass
        # print("Cannot find occupancy.")

    

@quantitativescore.register(StopsPickUp)
def _(transit_logics, data, scenario_num):
    if int(data["decision epoch"]) == transit_logics.request:
        pattern = re.compile(r"(\d+) stops before pick-up")
        text = data["stops"][:data["stops"].index("; ")]
        match = pattern.search(text)
        return int(match.group(1)) if match else 0

    elif int(data["decision epoch"]) < transit_logics.request:
        for child in data.get("children", []): 
            if int(child["assign to"][4:]) == transit_logics.vehicle:
                return quantitativescore(transit_logics, child, scenario_num)

    print("Cannot find anything with the specified tree depth.")
    return None



@quantitativescore.register(StopsDropOff)
def _(transit_logics, data, scenario_num):
    if int(data["decision epoch"]) == transit_logics.request:
        pattern = re.compile(r"(\d+) stops before drop-off")
        text = data["stops"][data["stops"].index("; ") + 2:]
        match = pattern.search(text)
        return int(match.group(1)) if match else 0

    elif int(data["decision epoch"]) < transit_logics.request:
        for child in data.get("children", []):
            if int(child["assign to"][4:]) == transit_logics.vehicle:
                return quantitativescore(transit_logics, child, scenario_num)

    print("Cannot find anything with the specified tree depth.")
    return None

    

@quantitativescore.register(Reward)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if car_number == transit_logics.node:
                return float(child["R"])
    # print("No match pattern found for reward.")



@quantitativescore.register(DecompReward1)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if car_number == transit_logics.node:
                return float(child["decomposed R"][0])
    # print("No match pattern found for DecompReward1.")



@quantitativescore.register(DecompReward2)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if car_number == transit_logics.node:
                return float(child["decomposed R"][1])
    # print("No match pattern found for DecompReward2.")



@quantitativescore.register(ETA)
def _(transit_logics, data, scenario_num):
    children = data["children"]
    pattern = r"Car (\d+)"
    for child in children:
        text = child["assign to"]
        match = re.search(pattern, text)
        if match:
            car_number = int(match.group(1))
            if car_number == transit_logics.request:
                result = child["eta"]
                if result == "Fulfilled":
                    return child["time"]
                else:
                    return result


@quantitativescore.register(DegreeVioDelay)
def _(transit_logics, data, scenario_num):
    request_time = quantitativescore(transit_logics.time, data, scenario_num)
    queried_vehicle = transit_logics.est_arrival.request
    times = enumerate_nodes(
        data, [], 
        checking_function=check_delay, 
        check_type=transit_logics.time,
        check_vehicle=queried_vehicle,
        time_window=scenario_num*5
    )
    return average_time_difference(times, request_time, type='d', time_window=scenario_num*5)

    

@quantitativescore.register(DegreeVioAdv)
def _(transit_logics, data, scenario_num):
    request_time = quantitativescore(transit_logics.time, data, scenario_num)
    queried_vehicle = transit_logics.est_arrival.request
    times = enumerate_nodes(
        data, [], 
        checking_function=check_delay, 
        check_type=transit_logics.time,
        check_vehicle=queried_vehicle,
        time_window=scenario_num*5
    )
    return average_time_difference(times, request_time, type='a', time_window=scenario_num*5)



@quantitativescore.register(ChanceVioDelay)
def _(transit_logics, data, scenario_num):
    request_time = quantitativescore(transit_logics.time, data, scenario_num)
    queried_vehicle = transit_logics.est_arrival.request
    delay_status = enumerate_nodes(
        data, [], 
        checking_function=bool_delay_status, 
        check_type=transit_logics.time,
        check_vehicle=queried_vehicle,
        reference_time=request_time,
        time_window=scenario_num*5
    )
    percentage = calculate_true_percentage(delay_status)
    return "{}%".format(percentage)



@quantitativescore.register(ChanceVioAdv)
def _(transit_logics, data, scenario_num):
    request_time = quantitativescore(transit_logics.time, data, scenario_num)
    queried_vehicle = transit_logics.est_arrival.request
    adv_status = enumerate_nodes(
        data, [], 
        checking_function=bool_adv_status, 
        check_type=transit_logics.time,
        check_vehicle=queried_vehicle,
        reference_time=request_time,
        time_window=scenario_num*5
    )
    percentage = calculate_true_percentage(adv_status)
    return "{}%".format(percentage)



@quantitativescore.register(CapacityVioQuant)
def _(transit_logics, data, scenario_num):
    return quantitativescore(transit_logics.cap, data, scenario_num) - quantitativescore(transit_logics.occ, data, scenario_num)


@quantitativescore.register(CompVioTime)
def _(transit_logics, data, scenario_num):
    return [quantitativescore(transit_logics.left, data, scenario_num), quantitativescore(transit_logics.right, data, scenario_num)]


@quantitativescore.register(CompPctTime)
def _(transit_logics, data, scenario_num):
    return [quantitativescore(transit_logics.left, data, scenario_num), quantitativescore(transit_logics.right, data, scenario_num)]


@quantitativescore.register(CompReward)
def _(transit_logics, data, scenario_num):
    try:
        return quantitativescore(transit_logics.left, data, scenario_num) - quantitativescore(transit_logics.right, data, scenario_num)
    except:
        "Fail to get information."

@quantitativescore.register(CompNumStops)
def _(transit_logics, data, scenario_num):
    try:
        return quantitativescore(transit_logics.left, data, scenario_num) - quantitativescore(transit_logics.right, data, scenario_num)
    except:
        "Fail to get information."




@singledispatch

def qualitativescore(transit_logics, data, scenario_num):
    raise TypeError("No qualitativescore for {} of class {}".format(transit_logics, transit_logics.__class__))
    
@qualitativescore.register(CapacityVio)
def _(transit_logics, data, scenario_num):
    return (quantitativescore(transit_logics.occ, data, scenario_num) - quantitativescore(transit_logics.cap, data, scenario_num)) >= 0

@qualitativescore.register(CompReward)
def _(transit_logics, data, scenario_num):
    return (quantitativescore(transit_logics.left, data, scenario_num) - quantitativescore(transit_logics.right, data, scenario_num)) < 0



@singledispatch

def getval(transit_logics, data, scenario_num):
    raise TypeError("No getval for {} of class {}".format(transit_logics, transit_logics.__class__))



def count_nodes(tree):
    if not tree.get("children", []):
        return 1
    
    total_count = 1
    for child in tree["children"]:
        total_count += count_nodes(child)
    
    return total_count



if __name__ == '__main__':
    # open data file. 

    file_path = 'backend/data/transit_additional_search/exp_large_4.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("File loaded.")
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Failed to decode JSON.")

    print("tree size:", count_nodes(data))

    # test the parser. 
    logic_test = [
        'tp(0)', 'td(0)', 'C(0)', 
        'O(1234,2)', 'O(1,1)',
        # 'vcv(C(2), O(1,2))', 
        # 'Phi3(r(2), r(1))',
        # 'Phi3(rd1(2), rd1(1))',
        # 'Phi3(rd2(2), rd2(1))', 
        'N(1,1)',
        # 'viod(td(0),eta(0))',
        # 'vioa(td(0),eta(0))', 
        # 'viod(tp(0),eta(0))',
        # 'viod(td(1),eta(1))',
        # 'vioa(td(1),eta(1))', 
        # 'viod(tp(1),eta(1))',
        # 'viod(td(2),eta(2))',
        # 'vioa(td(2),eta(2))', 
        # 'viod(tp(2),eta(2))',
        # 'viod(td(3),eta(3))',
        # 'vioa(td(3),eta(3))', 
        # 'viod(tp(3),eta(3))',
        # 'viod(td(4),eta(4))',
        # 'vioa(td(4),eta(4))', 
        # 'viod(tp(4),eta(4))',
        # 'viod(td(5),eta(5))',
        # 'vioa(td(5),eta(5))', 
        # 'viod(tp(5),eta(5))',
        # 'viod(td(6),eta(6))',
        # 'vioa(td(6),eta(6))', 
        # 'viod(tp(6),eta(6))',
        # 'viod(td(7),eta(7))',
        # 'vioa(td(7),eta(7))', 
        # 'viod(tp(7),eta(7))',
        # 'pctd(td(0),eta(0))',
        # 'pcta(td(0),eta(0))', 
        # 'pctd(tp(0),eta(0))',
        # 'pctd(td(1),eta(1))',
        # 'pcta(td(1),eta(1))', 
        # 'pctd(tp(1),eta(1))',
        # 'pctd(td(2),eta(2))',
        # 'pcta(td(2),eta(2))', 
        # 'pctd(tp(2),eta(2))',
        # 'pctd(td(3),eta(3))',
        # 'pcta(td(3),eta(3))', 
        # 'pctd(tp(3),eta(3))',
        # 'pctd(td(4),eta(4))',
        # 'pcta(td(4),eta(4))', 
        # 'pctd(tp(4),eta(4))',
        # 'pctd(td(5),eta(5))',
        # 'pcta(td(5),eta(5))', 
        # 'pctd(tp(5),eta(5))',
        # 'pctd(td(6),eta(6))',
        # 'pcta(td(6),eta(6))', 
        # 'pctd(tp(6),eta(6))',
        # 'pctd(td(7),eta(7))',
        # 'pcta(td(7),eta(7))', 
        # 'pctd(tp(7),eta(7))',
        # 'Phi3 ( r ( 1 ) , r ( 0 )  )',
        # 'Phi3 ( rd1 ( 1 ) , rd1 ( 0 )  )',
        # 'Phi3 ( rd2 ( 1 ) , rd2 ( 0 )  )', 
        # 'Phi3 ( r ( 3 ) , r ( 1 )  )',
        # 'Phi3 ( rd1 ( 3 ) , rd1 ( 1 )  )',
        # 'Phi3 ( rd2 ( 3 ) , rd2 ( 1 )  )', 
        # 'Phi3 ( r ( 3 ) , r ( 2 )  )',
        # 'Phi3 ( rd1 ( 3 ) , rd1 ( 2 )  )',
        # 'Phi3 ( rd2 ( 3 ) , rd2 ( 2 )  )', 
        # 'Phi3 ( r ( 3 ) , r ( 4 )  )',
        # 'Phi3 ( rd1 ( 3 ) , rd1 ( 4 )  )',
        # 'Phi3 ( rd2 ( 3 ) , rd2 ( 4 )  )', 
        'r(0)', 'rd1(0)', 'rd2(0)', 
        'r(1)', 
        # 'rd1(1)', 'rd2(1)', 
        # 'r(2)', 'rd1(2)', 'rd2(2)', 
        # 'r(3)', 'rd1(3)', 'rd2(3)', 
        # 'r(4)', 'rd1(4)', 'rd2(4)', 
        # 'r(5)', 'rd1(5)', 'rd2(5)', 
        # 'r(6)', 'rd1(6)', 'rd2(6)', 
        # 'r(7)', 'rd1(7)', 'rd2(7)', 
        # 'Phi1(vioa(tp(0),eta(0)), vioa(tp(1),eta(1)))',
        # 'Phi1(vioa(td(0),eta(0)), vioa(td(1),eta(1)))',
        # 'Phi1(vioa(tp(2),eta(2)), vioa(tp(0),eta(0)))',
        # 'Phi1(vioa(tp(1),eta(1)), vioa(tp(4),eta(4)))',
        # 'Phi2(pcta(tp(3),eta(3)), pcta(tp(1),eta(1)))',
        # 'Phi2(pcta(tp(2),eta(2)), pcta(tp(0),eta(0)))',
        # 'Phi2(pcta(tp(1),eta(1)), pcta(tp(4),eta(4)))',
        # 'Phi1(viod(td(3),eta(3)), viod(td(1),eta(1)))',
        # 'Phi1(viod(td(2),eta(2)), viod(td(0),eta(0)))',
        # 'Phi1(viod(td(1),eta(1)), viod(td(0),eta(0)))',
        # 'Phi1(viod(tp(1),eta(1)), viod(tp(0),eta(0)))',
        # 'Phi2(pctd(td(3),eta(3)), pctd(td(1),eta(1)))',
        # 'Phi2(pctd(td(2),eta(2)), pctd(td(0),eta(0)))',
        # 'Phi2(pctd(td(1),eta(1)), pctd(td(0),eta(0)))',
        # 'Phi2(pcta(tp(1),eta(1)), pcta(tp(0),eta(0)))',
        # 'Phi4(sp(2,0), sp(2,1))', 'Phi4(sd(2,0), sd(2,1))',
        # 'Phi4(sp(2,3), sp(2,4))', 'Phi4(sd(2,3), sd(2,4))',
        # 'Phi4(sp(2,0), sd(2,0))',
        # 'sp(2,0)', 'sd(2,0)', 'eta(1)'
    ]

    time_start = time.time()
    
    for i in range(5): 
        for logic in logic_test: 
            # print("Generated logic:", logic)
            parsed = parse(logic)
            # print("Quantitative/boolean result:")
            try:
                quantitativescore(parsed, data, 2)
            except TypeError:
                pass
            try:
                qualitativescore(parsed, data, 2)
            except TypeError:
                pass

        # print("\n")

    print(time.time() - time_start)
    print("\nDone.")