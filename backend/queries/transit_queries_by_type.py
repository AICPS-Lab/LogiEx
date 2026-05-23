import os
import sys
import json
import time
from openai import OpenAI



type_1 = [
    "Could you provide the designated pick-up time for the passenger?",
    "What is the appointed pick-up time for the passenger?",
    "Can you tell me the requested pick-up time for the passenger?",
    "Please inform me of the arranged pick-up time for the passenger.",
    "What time is the passenger supposed to be picked up?",
    "Could you confirm the set pick-up time for the passenger?",
    "What is the agreed-upon pick-up time for the passenger?",
    "Can you specify the pick-up time scheduled for the passenger?",
    "What time has been scheduled for picking up the passenger?",
    "Please provide the pick-up schedule for the passenger.",
    "What time is the passenger designated to be picked up?",
    "Could you tell me the expected pick-up time for the passenger?",
    "What pick-up time has been established for the passenger?",
    "Can you indicate the pick-up time for the passenger?",
    "Please disclose the scheduled pick-up time for the passenger.",
    "What is the determined pick-up time for the passenger?",
    "Could you ascertain the pick-up time for the passenger?",
    "Can you verify the pick-up time arranged for the passenger?",
    "What time will the passenger be picked up?",
    "Please confirm the exact pick-up time for the passenger.",
]


type_2 = [
    "Could you tell me the planned drop-off time for the passenger?",
    "What time has been set for the passenger's drop-off?",
    "Can you provide the arranged drop-off time for the passenger?",
    "Please inform me of the designated drop-off time for the passenger.",
    "What is the agreed-upon drop-off time for the passenger?",
    "Could you confirm the drop-off schedule for the passenger?",
    "What time is the passenger expected to be dropped off?",
    "Can you specify the drop-off time established for the passenger?",
    "Please disclose the time arranged for the passenger's drop-off.",
    "What is the official drop-off time for the passenger?",
    "What time is the passenger due to be dropped off?",
    "Could you provide the expected drop-off time for the passenger?",
    "What is the appointed time for the passenger's drop-off?",
    "Can you indicate when the passenger is scheduled to be dropped off?",
    "Please let me know the drop-off time set for the passenger.",
    "What time has been arranged for dropping off the passenger?",
    "Could you specify the drop-off time for the passenger?",
    "What is the confirmed time for the passenger's drop-off?",
    "Can you confirm the time the passenger will be dropped off?",
    "Please verify the scheduled time for the passenger's drop-off.",
]


type_3 = [
    "Could you tell me how many passengers are currently in vehicle {}?",
    "How many passengers are there on vehicle {} at the moment?",
    "Can you provide the current passenger count for vehicle {}?",
    "Please inform me of the number of passengers currently on vehicle {}.",
    "What is the current passenger count aboard vehicle {}?",
    "Could you specify the current number of passengers in vehicle {}?",
    "How many passengers are presently aboard vehicle {}?",
    "What's the count of passengers on vehicle {} right now?",
    "Can you confirm how many passengers are currently traveling in vehicle {}?",
    "Please verify the current number of passengers onboard vehicle {}.",
    "What's the current load of passengers in vehicle {}?",
    "Could you update me on how many passengers are currently in vehicle {}?",
    "What is the count of passengers currently traveling on vehicle {}?",
    "Can you report the number of passengers on board vehicle {} at this time?",
    "Please provide the current tally of passengers in vehicle {}.",
    "How full is vehicle {} with passengers right now?",
    "What number of passengers are we looking at in vehicle {} currently?",
    "Could you let me know the current headcount in vehicle {}?",
    "What does the passenger manifest for vehicle {} read at the moment?",
    "How many passengers does vehicle {} currently carry?"
]


type_4 = [
    "What is the current available seating capacity for vehicle {}?",
    "How many additional passengers can vehicle {} accommodate?",
    "Could you provide the remaining capacity for passengers in vehicle {}?",
    "What is the leftover passenger capacity in vehicle {}?",
    "Can you tell me how many more passengers vehicle {} can carry?",
    "What is the unused capacity for passengers on vehicle {}?",
    "Please inform me about the available capacity for additional passengers in vehicle {}.",
    "How much space is left for more passengers in vehicle {}?",
    "What is the maximum number of additional passengers that vehicle {} can accommodate?",
    "Could you specify the available passenger slots in vehicle {}?",
    "What is the available passenger capacity left in vehicle {}?",
    "How many seats are still available for passengers in vehicle {}?",
    "Could you tell me the remaining seating availability in vehicle {}?",
    "What is the number of open seats left for passengers in vehicle {}?",
    "Can you update me on the unoccupied passenger seats in vehicle {}?",
    "Please clarify the remaining passenger accommodation in vehicle {}.",
    "How many more seats can be filled in vehicle {}?",
    "What is the remaining passenger space in vehicle {}?",
    "Can vehicle {} still take more passengers? If so, how many?",
    "What is the capacity for additional passengers in vehicle {} right now?"
]


type_5 = [ 
    "What is the number of stops a passenger would encounter before reaching their destination if assigned to vehicle {}?",
    "How many stops are involved if a passenger is assigned to vehicle {}?",
    "Can you tell me the count of stops a passenger has to endure with vehicle {}?",
    "Please specify how many stops there will be for a passenger in vehicle {}.",
    "Could you inform me about the number of stops required for a passenger on vehicle {}?",
    "What are the total stops a passenger must wait through when assigned to vehicle {}?",
    "How many stops would be on the route for a passenger in vehicle {}?",
    "What is the stop count a passenger would face in vehicle {}?",
    "Could you detail the number of stops on the journey for a passenger traveling in vehicle {}?",
    "What's the estimated number of stops for a passenger if placed in vehicle {}?",
    "How many intermediate stops will there be for a passenger assigned to vehicle {}?",
    "Can you provide the stop total a passenger will experience when riding in vehicle {}?",
    "What is the estimated number of stops for a passenger when assigned to vehicle {}?",
    "Please list the number of stops a passenger would have if traveling in vehicle {}.",
    "Could you clarify how many stops are planned for a passenger in vehicle {}?",
    "How many stops should a passenger expect if they are assigned to vehicle {}?",
    "What is the expected number of transit stops for a passenger in vehicle {}?",
    "Can you enumerate the stops a passenger will go through on vehicle {}?",
    "What number of route stops will a passenger face in vehicle {}?",
    "How many stopping points are there for a passenger on vehicle {}?"
]


type_6 = [
    "Could you provide the estimated time of arrival for the passenger if assigned to vehicle {}?",
    "Can you give me the estimated arrival time for the passenger in vehicle {}?", 
    "What is the expected arrival time for the passenger if they are in vehicle {}?", 
    "Could you inform me of the ETA for the passenger when placed in vehicle {}?", 
    "Please tell me when the passenger is expected to arrive if they are in vehicle {}?",
    "Could you let me know the projected arrival time for the passenger aboard vehicle {}?", 
    "I need the arrival time estimation for the passenger assigned to vehicle {}." ,
    "What time should I expect the passenger to arrive if they travel in vehicle {}?",
    "Can you calculate the estimated time the passenger will arrive if using vehicle {}?", 
    "What would be the arrival time for the passenger if they take vehicle {}?",
    "Please provide the expected time of arrival for the passenger if traveling in vehicle {}.",
    "Could you determine the ETA for the passenger when traveling in vehicle {}?",
    "What's the anticipated time of arrival for the passenger in vehicle {}?",
    "Can you estimate the arrival time for the passenger if they're assigned to vehicle {}?",
    "Would you be able to tell me the arrival time for the passenger using vehicle {}?",
    "How soon will the passenger arrive if they are in vehicle {}?",
    "Please specify the projected time of arrival for the passenger in vehicle {}.",
    "Could you assess the arrival time for the passenger designated to vehicle {}?",
    "What is the ETA for the passenger when placed aboard vehicle {}?",
    "Can you forecast the time the passenger assigned to vehicle {} will arrive?",
    "Please supply the estimated time of arrival for the passenger aboard vehicle {}."
]


type_7 = [
    "How long is the delay expected to be when a passenger is assigned to vehicle {}?",
    "Can you tell me the anticipated delay time for assigning a passenger to vehicle {}?",
    "What duration of delay should we expect when a passenger boards vehicle {}?",
    "Could you estimate the delay length when a passenger is placed in vehicle {}?",
    "What is the typical delay for a passenger assigned to vehicle {}?",
    "Please provide the expected time of delay for a passenger using vehicle {}.",
    "How much delay can we anticipate when assigning a passenger to vehicle {}?",
    "Could you inform me about the delay duration expected with vehicle {}?",
    "What's the projected delay for passengers assigned to vehicle {}?",
    "Can you calculate the expected delay if a passenger is assigned to vehicle {}?",
    "What delay should be planned for when a passenger is assigned to vehicle {}?",
    "How extensive is the delay expected to be for a passenger in vehicle {}?",
    "What's the usual wait time when a passenger is assigned to vehicle {}?",
    "Could you specify the expected delay when using vehicle {} for a passenger?",
    "How significant is the expected delay for assigning a passenger to vehicle {}?",
    "Can you assess how long the passenger might be delayed in vehicle {}?",
    "What is the expected duration of the delay when a passenger travels in vehicle {}?",
    "Please estimate the delay a passenger would face in vehicle {}.",
    "How long will the passenger likely be delayed when placed in vehicle {}?",
    "What is the typical duration of delay for passengers assigned to vehicle {}?"
]


type_8 = [
    "How long is the advancement expected to take when a passenger is assigned to vehicle {}?", 
    "Can you tell me the anticipated advancement duration for assigning a passenger to vehicle {}?", 
    "What duration of advancement should we expect when a passenger boards vehicle {}?", 
    "Could you estimate the advancement length when a passenger is placed in vehicle {}?", 
    "What is the typical advancement duration for a passenger assigned to vehicle {}?", 
    "Please provide the expected time of advancement for a passenger using vehicle {}.", 
    "How much advancement can we anticipate when assigning a passenger to vehicle {}?", 
    "Could you inform me about the advancement duration expected with vehicle {}?", 
    "What's the projected advancement duration for passengers assigned to vehicle {}?", 
    "Can you calculate the expected advancement if a passenger is assigned to vehicle {}?", 
    "What advancement duration should be planned for when a passenger is assigned to vehicle {}?", 
    "How extensive is the advancement expected to be for a passenger in vehicle {}?", 
    "What's the usual advancement time when a passenger is assigned to vehicle {}?", 
    "Could you specify the expected advancement duration when using vehicle {} for a passenger?", 
    "How significant is the expected advancement duration for assigning a passenger to vehicle {}?", 
    "Can you assess how long the passenger might be advanced in vehicle {}?", 
    "What is the expected duration of the advancement when a passenger travels in vehicle {}?", 
    "Please estimate the advancement a passenger would face in vehicle {}.", 
    "How long will the passenger likely be advanced when placed in vehicle {}?", 
    "What is the typical duration of advancement for passengers assigned to vehicle {}?"
]


type_9 = [
    "What are the chances that the passenger will be delayed when assigned to vehicle {}?",
    "Can you calculate the likelihood of a late arrival for the passenger in vehicle {}?",
    "What is the likelihood that the passenger assigned to vehicle {} will arrive late?",
    "Could you estimate the probability of a delay for the passenger in vehicle {}?",
    "How likely is it that the passenger will not arrive on time in vehicle {}?",
    "What's the risk of the passenger arriving late when assigned to vehicle {}?",
    "Can you tell me the odds of the passenger being late in vehicle {}?",
    "How probable is it that assigning the passenger to vehicle {} will result in a late arrival?",
    "What is the chance of a late arrival for the passenger in vehicle {}?",
    "Could you determine the probability of tardiness for the passenger in vehicle {}?",
    "What are the expected odds of the passenger arriving late in vehicle {}?",
    "Can you assess the probability that the passenger in vehicle {} will be delayed?",
    "How likely is the passenger to arrive late if they are assigned to vehicle {}?",
    "What is the estimated chance of a delay for the passenger when using vehicle {}?",
    "Can you provide the likelihood that the passenger will face a delay in vehicle {}?",
    "What percentage chance is there of the passenger arriving late to vehicle {}?",
    "Could you calculate the risk of lateness for the passenger in vehicle {}?",
    "How probable is a late arrival for the passenger when assigned to vehicle {}?",
    "What is the expected probability of the passenger being delayed in vehicle {}?",
    "Can you specify the chances of a late arrival for the passenger assigned to vehicle {}?"
]


type_10 = [
    "What are the chances that the passenger will arrive early if assigned to vehicle {}?",
    "Can you calculate the likelihood of the passenger being ahead of schedule in vehicle {}?",
    "What is the likelihood that the passenger assigned to vehicle {} will arrive ahead of time?",
    "Could you estimate the probability of early arrival for the passenger in vehicle {}?",
    "How likely is it that the passenger will be early in vehicle {}?",
    "What's the probability of the passenger arriving earlier than planned when assigned to vehicle {}?",
    "Can you tell me the odds of the passenger being ahead of schedule in vehicle {}?",
    "How probable is it that assigning the passenger to vehicle {} will result in an early arrival?",
    "What is the chance of an early arrival for the passenger in vehicle {}?",
    "Could you determine the probability of being ahead of schedule for the passenger in vehicle {}?",
    "What are the expected odds of the passenger arriving early in vehicle {}?",
    "Can you assess the probability that the passenger in vehicle {} will be early?",
    "How likely is the passenger to be ahead of schedule if they are assigned to vehicle {}?",
    "What is the estimated chance of early arrival for the passenger when using vehicle {}?",
    "Can you provide the likelihood that the passenger will face early arrival in vehicle {}?",
    "What percentage chance is there of the passenger being ahead of schedule to vehicle {}?",
    "Could you calculate the risk of earliness for the passenger in vehicle {}?",
    "How probable is an early arrival for the passenger when assigned to vehicle {}?",
    "What is the expected probability of the passenger arriving early in vehicle {}?",
    "Can you specify the chances of an early arrival for the passenger assigned to vehicle {}?"
]



type_11 = [
    "What causes the delay when assigning a passenger to vehicle {}?", 
    "Can you explain why there are delays with vehicle {} when it is assigned a passenger?",
    "What factors contribute to the delay of passenger assignments to vehicle {}?",
    "Why does vehicle {} experience delays when passengers are assigned to it?",
    "What are the contributing factors to delays when a passenger is allocated to vehicle {}?",
    "Can you identify the reasons for passenger delays in vehicle {}?",
    "What leads to the delay when a passenger is placed in vehicle {}?",
    "What circumstances cause delays for passengers assigned to vehicle {}?",
    "Why is there a delay when assigning passengers to vehicle {}?",
    "What issues cause delays when vehicle {} is assigned a passenger?",
    "What are the underlying factors causing delays with vehicle {} when it has a passenger assignment?",
    "Could you detail the causes for delays when a passenger is assigned to vehicle {}?",
    "What are the main contributors to the delay experienced by passengers in vehicle {}?",
    "Can you specify what triggers the delays for vehicle {} with passenger assignments?",
    "What typically causes the delays when passengers are designated to vehicle {}?",
    "What issues lead to a passenger's delay in vehicle {}?",
    "What factors are responsible for causing delays when vehicle {} is assigned passengers?",
    "Why do passengers assigned to vehicle {} frequently experience delays?",
    "Can you outline the reasons behind the delay occurrences in vehicle {} when assigned a passenger?",
    "What typically leads to delays for vehicle {} when tasked with transporting passengers?"
]



type_12 = [
    "Why was vehicle {} favored instead of vehicle {} for this particular task?",
    "Can you explain the decision to select vehicle {} rather than vehicle {} for this assignment?",
    "What factors influenced the choice of vehicle {} over vehicle {} in this instance?",
    "Could you detail the rationale behind assigning vehicle {} instead of vehicle {}?",
    "What led to the selection of vehicle {} over vehicle {} for this operation?",
    "Please provide the reasoning for preferring vehicle {} to vehicle {} for this duty.",
    "What considerations resulted in the choice of vehicle {} instead of vehicle {}?",
    "Why did vehicle {} get the nod over vehicle {} for this particular assignment?",
    "What were the deciding factors that led to choosing vehicle {} over vehicle {}?",
    "Could you clarify why vehicle {} was chosen over vehicle {} for the assignment?",
    "Can you explain the decision to select vehicle {} instead of vehicle {} for this task?",
    "What factors led to the preference of vehicle {} over vehicle {} in this assignment?",
    "Could you detail the reasons for choosing vehicle {} rather than vehicle {} for this operation?",
    "Why was vehicle {} favored over vehicle {} for this particular assignment?",
    "What motivated the selection of vehicle {} over vehicle {} for this duty?",
    "Please provide the justification for assigning vehicle {} instead of vehicle {}.",
    "What considerations influenced the choice of vehicle {} over vehicle {} for this role?",
    "Why did vehicle {} get the nod over vehicle {} for this project?",
    "What were the determining factors in the decision to use vehicle {} rather than vehicle {}?",
    "Could you specify the rationale behind the preference for vehicle {} over vehicle {} in this context?"
]



type_13 = [
    "Why did the algorithm not consider alternative vehicle {}?",
    "What were the reasons behind the algorithm's decision to exclude vehicle {}?",
    "Can you explain why vehicle {} was not considered by the algorithm?",
    "What factors contributed to the algorithm overlooking vehicle {}?",
    "Could you clarify what led to vehicle {} being disregarded by the algorithm?",
    "What caused the algorithm to bypass vehicle {} in its selection process?",
    "Why was vehicle {} not included in the algorithm's vehicle consideration set?",
    "Can you detail the considerations that led to omitting vehicle {} from the algorithm's choices?",
    "What were the determining factors for the algorithm not to consider vehicle {}?",
    "Why did vehicle {} fail to be an option for the algorithm?",
    "What rationale did the algorithm have for not selecting vehicle {}?",
    "Can you specify why vehicle {} was excluded from the algorithm's evaluation?",
    "What led the algorithm to ignore vehicle {} in its decision-making?",
    "Could you provide insights into why vehicle {} was not considered a viable option by the algorithm?",
    "What were the key factors that led to vehicle {} being excluded by the algorithm?",
    "Why did the algorithm decide against considering vehicle {}?",
    "Can you discuss the reasons for the algorithm's exclusion of vehicle {}?",
    "What influenced the algorithm's decision to leave out vehicle {}?",
    "Why was vehicle {} not factored into the algorithm's calculations?",
    "What motivated the algorithm's choice to disregard vehicle {}?"
]



type_14 = [
    "What could result from assigning the passenger to alternative vehicle {}?",
    "Can you explain the potential impacts of using vehicle {} for the passenger?",
    "What might be the outcomes if the passenger is assigned to vehicle {}?",
    "Could you describe the consequences of placing the passenger in vehicle {}?",
    "What are the expected effects of assigning the passenger to vehicle {}?",
    "What would happen if the passenger were assigned to vehicle {}?",
    "Can you detail what would follow from putting the passenger in vehicle {}?",
    "What are the possible repercussions of assigning the passenger to vehicle {}?",
    "What could be the fallout of using vehicle {} for the passenger?",
    "Could you outline the potential issues with assigning the passenger to vehicle {}?",
    "What might go wrong if the passenger is assigned to vehicle {}?",
    "What are the risks of placing the passenger in vehicle {}?",
    "Can you speculate on the impact of assigning the passenger to vehicle {}?",
    "What negative outcomes might arise from using vehicle {} for the passenger?",
    "What would the implications be for assigning the passenger to vehicle {}?",
    "Could assigning the passenger to vehicle {} lead to any problems?",
    "What are potential drawbacks of assigning the passenger to vehicle {}?",
    "Could you project the consequences if the passenger is assigned to vehicle {}?",
    "What would the effect be of putting the passenger in vehicle {}?",
    "What possible issues could assigning the passenger to vehicle {} bring about?"
]



type_15 = [
    "What benefits does vehicle {} offer over vehicle {} when capacity isn't an issue?",
    "Can you highlight the advantages of vehicle {} compared to vehicle {} without considering capacity?",
    "What are the key benefits of vehicle {} over vehicle {} if we disregard capacity?",
    "Could you explain the superior aspects of vehicle {} over vehicle {} when capacity is excluded?",
    "What makes vehicle {} better than vehicle {} if we ignore capacity concerns?",
    "What are the strengths of vehicle {} compared to vehicle {} if capacity is not factored in?",
    "Can you detail the positives of using vehicle {} instead of vehicle {} when capacity isn't a factor?",
    "What advantages does vehicle {} hold over vehicle {} when not looking at capacity?",
    "Why is vehicle {} preferred over vehicle {} if capacity is not a concern?",
    "What are the compelling reasons to choose vehicle {} over vehicle {} when ignoring capacity?",
    "Can you outline the benefits that vehicle {} has over vehicle {} with capacity out of the equation?",
    "What superior features does vehicle {} offer compared to vehicle {} if capacity doesn't matter?",
    "Could you specify the advantages of vehicle {} against vehicle {} when capacity isn't considered?",
    "What makes vehicle {} more advantageous than vehicle {} without considering capacity limitations?",
    "Why might vehicle {} be a better choice than vehicle {} if we overlook capacity?",
    "Can you discuss the merits of sticking with vehicle {} over switching to vehicle {} if capacity isn't an issue?",
    "What are the advantages of maintaining vehicle {} over opting for vehicle {} with no capacity concerns?",
    "What beneficial aspects does vehicle {} have over vehicle {} when capacity is not the priority?",
    "Could you elaborate on why vehicle {} is favored over vehicle {} when capacity is not restrictive?",
    "What positive outcomes might result from vehicle {} compared to vehicle {} if capacity concerns are set aside?"
]



type_16 = [
    "Does vehicle {} reduce time violations more effectively than vehicle {}?",
    "Is vehicle {} better at minimizing time violations compared to vehicle {}?",
    "Can you confirm if vehicle {} has fewer time violations than vehicle {}?",
    "Does vehicle {} outperform vehicle {} in minimizing time violations?",
    "Is there a difference in time violation rates between vehicle {} and vehicle {}?",
    "Can vehicle {} be considered superior to vehicle {} for reducing time violations?",
    "How does vehicle {} compare to vehicle {} in terms of time violation minimization?",
    "Does vehicle {} have more time violations than vehicle {}?",
    "In managing time violations, does vehicle {} excel over vehicle {}?",
    "Is vehicle {} more effective than vehicle {} at avoiding time violations?",
    "Which performs better in minimizing time violations: vehicle {} or vehicle {}?",
    "Can you detail how vehicle {} performs against vehicle {} in reducing time violations?",
    "Is vehicle {} less prone to time violations than vehicle {}?",
    "Does vehicle {} show superiority in time violation reduction compared to vehicle {}?",
    "How effective is vehicle {} versus vehicle {} in minimizing time violations?",
    "Does vehicle {} manage time violations better than vehicle {}?",
    "Are time violations less frequent with vehicle {} compared to vehicle {}?",
    "Can you compare the effectiveness in minimizing time violations between vehicle {} and vehicle {}?",
    "Does vehicle {} have a better track record in minimizing time violations than vehicle {}?",
    "In terms of minimizing time violations, is vehicle {} more advantageous than vehicle {}?"
]



type_17 = [
    "Is the number of stops lower for passengers in vehicle {} than in vehicle {}?",
    "Will passengers in vehicle {} encounter fewer stops than those in vehicle {}?",
    "Does vehicle {} offer a ride with fewer stops compared to vehicle {}?",
    "Can you confirm if vehicle {} has fewer stops than vehicle {} for passengers?",
    "Are there fewer stops for passengers traveling in vehicle {} compared to those in vehicle {}?",
    "Does taking vehicle {} result in fewer stops than vehicle {}?",
    "Will traveling in vehicle {} reduce the number of stops compared to vehicle {}?",
    "Is vehicle {} better in terms of having fewer stops for passengers than vehicle {}?",
    "Can vehicle {} provide a smoother journey with fewer stops than vehicle {}?",
    "Will the ride in vehicle {} include fewer stops than in vehicle {}?",
    "Are passengers likely to face fewer stops in vehicle {} as opposed to vehicle {}?",
    "Does vehicle {} ensure a lesser number of stops compared to vehicle {}?",
    "Will the passenger journey in vehicle {} have fewer stops compared to that in vehicle {}?",
    "Could passengers expect fewer stops with vehicle {} over vehicle {}?",
    "Is it true that vehicle {} will have fewer passenger stops than vehicle {}?",
    "Will the trip in vehicle {} have fewer stops compared to the trip in vehicle {}?",
    "Does choosing vehicle {} over vehicle {} reduce the number of stops for passengers?",
    "Are fewer stops anticipated for vehicle {} as compared to vehicle {}?",
    "Can we expect the route for vehicle {} to have fewer stops than that for vehicle {}?",
    "Is the routing for vehicle {} designed with fewer stops compared to vehicle {}?"
]



def load_query_by_type(type):
    type_dict = {
        1: type_1, 2: type_2, 3: type_3,
        4: type_4, 5: type_5, 6: type_6,
        7: type_7, 8: type_8, 9: type_9,
        10: type_10, 11: type_11,
        12: type_12, 13: type_13,
        14: type_14, 15: type_15,
        16: type_16, 17: type_17
    }
    return type_dict.get(type)




def main(): 
    print("Done.")

if __name__ == "__main__":
    main()