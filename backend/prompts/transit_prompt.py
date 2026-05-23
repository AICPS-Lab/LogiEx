standard_prompt_type_1 = '''
Input: Can you tell me the scheduled pick-up time for the passenger?
Answer: tp(0)
Input: Would you please inform me about the planned pick-up time for the passenger?
Answer: tp(0)
Input: Could you specify the arranged pick-up time for the passenger?
Answer: tp(0)
Input: {input}
'''




standard_prompt_type_2 = '''
Input: Could you tell me when the passenger is scheduled to be dropped off?
Answer: td(0)
Input: I need to know the scheduled drop-off time for the passenger.
Answer: td(0)
Input: Can you provide the designated drop-off time for the passenger?
Answer: td(0)
Input: {input}
'''


standard_prompt_type_3 = '''
Input: Can you provide the current passenger count for vehicle 1?
Answer: O(0,1)
Input: Would you please inform me of the number of passengers presently on vehicle 2?
Answer: O(0,2)
Input: Could you specify the total number of passengers currently aboard vehicle 3?
Answer: O(0,3)
Input: {input}
'''



standard_prompt_type_4 = '''
Input: What is the remaining passenger capacity of vehicle 2?
Answer: vcvq(C(2), O(1,2))
Input: How many additional passengers can vehicle 1 accommodate?
Answer: vcvq(C(1), O(1,1))
Input: What is the maximum number of extra passengers that can fit in vehicle 3?
Answer: vcvq(C(3), O(1,3))
Input: {input}
'''



standard_prompt_type_5 = '''
Input: What is the number of stops a passenger will face when traveling in vehicle 1?
Answer: sp(0,1); sd(0,1)
Input: Can you specify how many stops there are for a passenger assigned to vehicle 2?
Answer: sp(0,2); sd(0,2)
Input: Would you inform me about the total stops on the route for a passenger in vehicle 3?
Answer: sp(0,3); sd(0,3)
Input: {input}
'''





standard_prompt_type_6 = '''
Input: Could you indicate the expected arrival time for the passenger if they're assigned to vehicle 1?
Answer: eta(1)
Input: Please calculate and inform me of the passenger's estimated arrival time in vehicle 2.
Answer: eta(2)
Input: Can you tell me what time the passenger is likely to arrive if they travel in vehicle 3?
Answer: eta(3)
Input: {input}
'''



standard_prompt_type_7 = '''
Input: Could you let me know the likely delay duration in the pick-up time for a passenger assigned to vehicle 3?
Answer: viod(tp(3),eta(3))
Input: What length of delay in the pick-up time is anticipated for passengers traveling in vehicle 2?
Answer: viod(tp(2),eta(2))
Input: Can you provide an estimate of how long the delay might be in the pick-up time when a passenger is allocated to vehicle 1?
Answer: viod(tp(1),eta(1))
Input: {input}
'''



standard_prompt_type_8 = '''
Input: Could you let me know the likely delay duration in the drop-off time for a passenger assigned to vehicle 3?
Answer: viod(td(3),eta(3))
Input: What length of delay in the drop-off time is anticipated for passengers traveling in vehicle 2?
Answer: viod(td(2),eta(2))
Input: Can you provide an estimate of how long the delay might be in the drop-off time when a passenger is allocated to vehicle 1?
Answer: viod(td(1),eta(1))
Input: {input}
'''



standard_prompt_type_9 = '''
Input: Could you clarify the expected time of advancement in the pick-up time for a passenger when they are assigned to vehicle 3?
Answer: vioa(tp(3),eta(3))
Input: What is the forecasted duration of advancement in the pick-up time for passengers in vehicle 2?
Answer: vioa(tp(2),eta(2))
Input: Can you provide details on how long it will take to advance a passenger in the pick-up time when assigned to vehicle 1?
Answer: vioa(tp(1),eta(1))
Input: {input}
'''



standard_prompt_type_10 = '''
Input: Could you clarify the expected time of advancement in the drop-off time for a passenger when they are assigned to vehicle 3?
Answer: vioa(td(3),eta(3))
Input: What is the forecasted duration of advancement in the drop-off time for passengers in vehicle 2?
Answer: vioa(td(2),eta(2))
Input: Can you provide details on how long it will take to advance a passenger in the drop-off time when assigned to vehicle 1?
Answer: vioa(td(1),eta(1))
Input: {input}
'''



standard_prompt_type_11 = '''
Input: What is the expected rate of delay for picking up the passenger when traveling in vehicle 1?
Answer: pctd(tp(1),eta(1))
Input: Can you quantify the chance that the passenger will be picked up later than scheduled in vehicle 2?
Answer: pctd(tp(2),eta(2))
Input: How significant is the likelihood that the passenger will not be picked up on time if placed in vehicle 3?
Answer: pctd(tp(3),eta(3))
Input: {input}
'''



standard_prompt_type_12 = '''
Input: What is the expected rate of delay for dropping off the passenger when traveling in vehicle 1?
Answer: pctd(td(1),eta(1))
Input: Can you quantify the chance that the passenger will be dropped off later than scheduled in vehicle 2?
Answer: pctd(td(2),eta(2))
Input: How significant is the likelihood that the passenger will not be dropped off on time if placed in vehicle 3?
Answer: pctd(td(3),eta(3))
Input: {input}
'''


standard_prompt_type_13 = '''
Input: How likely is it that the passenger will be picked up ahead of schedule in vehicle 1?
Answer: pcta(tp(1),eta(1))
Input: Can you estimate the probability of the passenger being picked up before the expected time in vehicle 2?
Answer: pcta(tp(2),eta(2))
Input: What is the possibility that the passenger will be early to their pick up with vehicle 3?
Answer: pcta(tp(3),eta(3))
Input: {input}
'''



standard_prompt_type_14 = '''
Input: How likely is it that the passenger will reach their destination ahead of schedule in vehicle 1?
Answer: pcta(td(1),eta(1))
Input: Can you estimate the probability of the passenger arriving before the expected time in vehicle 2?
Answer: pcta(td(2),eta(2))
Input: What is the possibility that the passenger will be early to their destination with vehicle 3?
Answer: pcta(td(3),eta(3))
Input: {input}
'''



standard_prompt_type_15 = '''
Input: What factors lead to delays when a passenger is assigned to vehicle 1?
Answer: sp(0,1); sd(0,1)
Input: Can you identify the reasons for the delay associated with assigning a passenger to vehicle 2?
Answer: sp(0,2); sd(0,2)
Input: Why does assigning a passenger to vehicle 3 result in delays?
Answer: sp(0,3); sd(0,3)
Input: {input}
'''



standard_prompt_type_16 = '''
Input: What were the reasons for choosing vehicle 2 over vehicle 3 for this assignment?
Answer: vcv(C(3), O(1,3)); Phi3(r(2), r(3)); Phi3(rd1(2), rd1(3)); Phi3(rd2(2), rd2(3))
Input: Can you explain the rationale behind selecting vehicle 1 instead of vehicle 2 for this task?
Answer: vcv(C(2), O(1,2)); Phi3(r(1), r(2)); Phi3(rd1(1), rd1(2)); Phi3(rd2(1), rd2(2))
Input: What led to the decision to use vehicle 3 rather than vehicle 4 for this operation?
Answer: vcv(C(4), O(1,4)); Phi3(r(3), r(4)); Phi3(rd1(3), rd1(4)); Phi3(rd2(3), rd2(4))
Input: {input}
'''


standard_prompt_type_17 = '''
Input: What led to the algorithm's decision to overlook vehicle 1?
Answer: vcv(C(1), O(1,1))
Input: Can you explain the reasons behind the algorithm's exclusion of vehicle 2?
Answer: vcv(C(2), O(1,2))
Input: Why was vehicle 3 not factored into the algorithm's considerations?
Answer: vcv(C(3), O(1,3))
Input: {input}
'''



standard_prompt_type_18 = '''
Input: What are the potential consequences of placing the passenger in alternative vehicle 1?
Answer: search(1)
Input: What might happen if the passenger is assigned to vehicle 2?
Answer: search(2)
Input: Can you describe the expected outcomes of using vehicle 3 for the passenger?
Answer: search(3)
Input: {input}
'''



standard_prompt_type_19 = '''
Input: How does vehicle 1 outperform vehicle 2 when capacity constraints are not a factor?
Answer: Phi1(vioa(tp(1),eta(1)), vioa(tp(2),eta(2))); Phi1(vioa(td(1),eta(1)), vioa(td(2),eta(2))); Phi1(viod(tp(1),eta(1)), viod(tp(2),eta(2))); Phi1(viod(td(1),eta(1)), viod(td(2),eta(2))); Phi4(sp(0,1), sp(0,2)); Phi4(sd(0,1), sd(0,2))
Input: What advantages does vehicle 3 have compared to vehicle 4 if we disregard capacity limitations?
Answer: Phi1(vioa(tp(3),eta(3)), vioa(tp(4),eta(4))); Phi1(vioa(td(3),eta(3)), vioa(td(4),eta(4))); Phi1(viod(tp(3),eta(3)), viod(tp(4),eta(4))); Phi1(viod(td(3),eta(3)), viod(td(4),eta(4))); Phi4(sp(0,3), sp(0,4)); Phi4(sd(0,3), sd(0,4))
Input: In what ways is vehicle 2 superior to vehicle 3 when ignoring capacity issues?
Answer: Phi1(vioa(tp(2),eta(2)), vioa(tp(3),eta(3))); Phi1(vioa(td(2),eta(2)), vioa(td(3),eta(3))); Phi1(viod(tp(2),eta(2)), viod(tp(3),eta(3))); Phi1(viod(td(2),eta(2)), viod(td(3),eta(3))); Phi4(sp(0,2), sp(0,3)); Phi4(sd(0,2), sd(0,3))
Input: {input}
'''




standard_prompt_type_20 = '''
Input: Is vehicle 1 more successful at minimizing time violations compared to vehicle 2?
Answer: Phi1(vioa(tp(1),eta(1)), vioa(tp(2),eta(2))); Phi1(vioa(td(1),eta(1)), vioa(td(2),eta(2))); Phi1(viod(tp(1),eta(1)), viod(tp(2),eta(2))); Phi1(viod(td(1),eta(1)), viod(td(2),eta(2)))
Input: Can vehicle 3 better prevent time violations than vehicle 4?
Answer: Phi1(vioa(tp(3),eta(3)), vioa(tp(4),eta(4))); Phi1(vioa(td(3),eta(3)), vioa(td(4),eta(4))); Phi1(viod(tp(3),eta(3)), viod(tp(4),eta(4))); Phi1(viod(td(3),eta(3)), viod(td(4),eta(4)))
Input: Does vehicle 2 have a lower rate of time violations than vehicle 3?
Answer: Phi1(vioa(tp(2),eta(2)), vioa(tp(3),eta(3))); Phi1(vioa(td(2),eta(2)), vioa(td(3),eta(3))); Phi1(viod(tp(2),eta(2)), viod(tp(3),eta(3))); Phi1(viod(td(2),eta(2)), viod(td(3),eta(3)))
Input: {input}
'''




standard_prompt_type_21 = '''
Input: Does vehicle 1 offer a route with fewer stops than vehicle 2?"
Answer: Phi4(sp(0,1), sp(0,2)); Phi4(sd(0,1), sd(0,2))
Input: Will traveling in vehicle 3 result in fewer stops compared to using vehicle 4?"
Answer: Phi4(sp(0,3), sp(0,4)); Phi4(sd(0,3), sd(0,4))
Input: Can passengers expect less frequent stopping when assigned to vehicle 2 instead of vehicle 3?
Answer: Phi4(sp(0,2), sp(0,3)); Phi4(sd(0,2), sd(0,3))
Input: {input}
'''



standard_prompt_type_22 = '''
Input: What does occur when traffic becomes congested? 
Answer: Cong(0)
Input: What happens in cases of heavy traffic?
Answer: Cong(0)
Input: What takes place during traffic congestion?
Answer: Cong(0)
Input: {input}
'''



standard_prompt_type_23 = '''
Input: What happens if vehicle 1 breaks down?
Answer: exclude(1)
Input: What should we do if vehicle 2 is out of service?
Answer: exclude(2)
Input: What is the plan if vehicle 3 cannot be used?
Answer: exclude(3)
Input: {input}
'''



standard_prompt_type_24 = '''
Input: What should we do if this trip includes 2 passengers?
Answer: multi(2)
Input: How do we manage if there are 3 passengers on this trip?
Answer: multi(3)
Input: How does the trip change if there are 4 passengers involved?
Answer: multi(4)
Input: {input}
'''



standard_prompt_type_25 = '''
Input: What is the reassignment plan for passengers currently on vehicle 2 if it breaks down? 
Answer: reassign(2)
Input: What alternative vehicle will be arranged for passengers aboard vehicle 1 in the event of a breakdown?
Answer: reassign(1)
Input: If vehicle 3 becomes inoperable, where will the passengers be reassigned? 
Answer: reassign(3)
Input: {input}
'''


standard_prompt_type_26 = '''
Input: Is there a possibility of a delay for the passenger if they are assigned to vehicle 3?
Answer: viod(tp(3),eta(3)); viod(td(3),eta(3))
Input: Is a delay expected for the passenger if they are assigned to vehicle 1?
Answer: viod(tp(1),eta(1)); viod(td(1),eta(1))
Input: Could the passenger face a delay if they are assigned to vehicle 2?
Answer: viod(tp(2),eta(2)); viod(td(2),eta(2))
Input: {input}
'''



standard_prompt_type_27 = '''
Input: Is it possible that the passenger will arrive earlier than expected if assigned to vehicle 3?
Answer: vioa(tp(3),eta(3)); vioa(td(3),eta(3))
Input: Is the passenger likely to arrive ahead of time if assigned to vehicle 2?
Answer: vioa(tp(2),eta(2)); vioa(td(2),eta(2))
Input: Could assigning the passenger to vehicle 1 result in them arriving too early?
Answer: vioa(tp(1),eta(1)); vioa(td(1),eta(1))
Input: {input}
'''


standard_prompt_type_28 = '''
Input: What is the difference in reward when the passenger is assigned to vehicle 1 versus vehicle 2?
Answer: Phi3(r(1), r(2)); Phi3(rd1(1), rd1(2)); Phi3(rd2(1), rd2(2))
Input: How does the reward change between assigning the passenger to vehicle 2 and vehicle 3?
Answer: Phi3(r(2), r(3)); Phi3(rd1(2), rd1(3)); Phi3(rd2(2), rd2(3))
Input: What are the reward variations when comparing assignments to vehicle 0 and vehicle 4?
Answer: Phi3(r(0), r(4)); Phi3(rd1(0), rd1(4)); Phi3(rd2(0), rd2(4))
Input: {input}
'''



standard_prompt_type_29 = '''
Input: Why was vehicle 1 chosen for the passenger's assignment?
Answer: vcv(C(1), O(1,1)); r(1); rd1(1); rd2(1)
Input: What led to the passenger being assigned to vehicle 2?
Answer: vcv(C(2), O(1,2)); r(2); rd1(2); rd2(2)
Input: What factors determined the assignment of the passenger to vehicle 3?
Answer: vcv(C(3), O(1,3)); r(3); rd1(3); rd2(3)
Input: {input}
'''


standard_prompt_type_30 = '''
Input: Which vehicle is scheduled to pick up the passenger?
Answer: car(1)
Input: Which vehicle is assigned to fulfill the passenger's trip request?
Answer: car(1)
Input: Which vehicle will be handling the passenger's trip request?
Answer: car(1)
Input: {input}
'''



standard_prompt_type_31 = '''
Input: How many vehicles are available right now to pick up the passenger?
Answer: availablecar(1)
Input: How many vehicles can pick up the passenger at this moment?
Answer: availablecar(1)
Input: What is the number of vehicles available to pick up the passenger at the moment?
Answer: availablecar(1)
Input: {input}
'''