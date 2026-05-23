import os
import sys
import numpy as np

## CTL model checker for timed discrete branching states
## 8 basic CTL operators: AX, EX, AG, EG, AF, EF, AU, EU

FILE_FOLD = "backend/algo/data/"
travel_time_matrix = np.loadtxt(os.path.join(FILE_FOLD,"travel_time_matrix/travel_time_matrix.csv"), delimiter=",")
travel_time_matrix = travel_time_matrix.astype(int)



def get_vehicle_id_in_list(v_list, v_id):
    for i in range(len(v_list)):
        if v_list[i].id == v_id:
            return i
    return None



def print_tree(node, indent=0):
    "A helper class to print and visualize a tree"
    "each node has a data from request status"

    def get_node_label(R):
        labels_count = [0, 0, 0, 0]
        status = ['waiting', 'assigned', 'in_transit', 'dropped_off']
        short = ['W', 'A', 'T', 'D']
        for r in R:
            labels_count[status.index(r.current_status.name)] += 1
        ret = [st+str(num) for num, st in zip(labels_count, short)]
        return '; '.join(ret)
    
    if not node.children:
        print(str("|---"*indent + get_node_label(node.data.R)))
    
    else:
        print(str("|---"*indent + get_node_label(node.data.R)))
        for child in node.children:
            print_tree(child, indent+1)




def print_tree_attr(node, indent=0):
    "A helper class to print and visualize a tree"
    "each node has a data from request status"
    "attr stores the result for CTL checking"

    def get_node_label(checking_result):
        if len(checking_result) > 0:
            return str('\n'.join(checking_result))
        else:
            return "No Violation"
    
    print(str("|---"*indent + get_node_label(node.checked)))
    if node.children:
        for child in node.children:
            print_tree_attr(child, indent+1)




def print_simple_tree(node, indent=0):
    "A helper class to print and visualize a tree"
    if not node.children:
        print(str("|---"*indent + node.data))
    else:
        print(str("|---"*indent + node.data))
        for child in node.children:
            print_simple_tree(child, indent+1)



def build_ctl_tree(tree, curr_node=None, current_idx=0):

    if curr_node is None:
        # Getting data for the root node.
        curr_node = next(iter(tree.keys()))  
        tree_node = CTLNode(0, curr_node)

        for index,child_node in enumerate(tree[curr_node]):
            child_node = build_ctl_tree(tree, child_node, index+current_idx)
            tree_node.add_child(child_node)
    
    else:
        tree_node = CTLNode(current_idx, curr_node)
        
        if curr_node in tree:
            for index,child_node in enumerate(tree[curr_node]):
                child_node = build_ctl_tree(tree, child_node, index+current_idx)
                tree_node.add_child(child_node)
    
    return tree_node



class CTLNode:
    """
    A CTL node class.
    """
    def __init__(self, idx, data, children=None):
        self.idx = idx
        self.data = data    # route plan object
        self.children = children if children else []
        

    def add_child(self, child):
        self.children.append(child)


    def update_data(self, new_data):
        "Update the data for a CTL tree node."
        self.data = new_data
        

    def evaluate(self, formula=" ", criterion="status"):
        """
        Evaluate the CTL node for a particular formula.
        A formula is a string 
        """
        if formula[0] == "not":
            return not self.evaluate(formula[1], criterion)

        elif formula[0] == "AX":
            return self.forall_next(self, formula[1], criterion)
        
        elif formula[0] == "EX":
            return self.exists_next(self, formula[1], criterion)
        
        elif formula[0] == "AF":
            return self.forall_future(self, formula[1], criterion)
        
        elif formula[0] == "EF":
            return self.exists_future(self, formula[1], criterion)
        
        elif formula[0] == "AG":
            return self.forall_global(self, formula[1], criterion)
        
        elif formula[0] == "EG":
            return self.exists_global(self, formula[1], criterion)
        
        elif formula[0] == "AU":    #TODO
            subformulas = formula[1]
            return self.forall_until(self, subformulas[0], subformulas[1], criterion)
        
        elif formula[0] == "EU":
            subformulas = formula[1]
            return self.exists_until(self, subformulas[0], subformulas[1], criterion)
        
        else:
            if criterion == "status":
                return self.eval_status(self.data, formula)
            elif criterion == "time":
                return self.time_hard_constraint()



    def time_hard_constraint(self, window=0):
        "`data` is a route plan object."
        pass
            

    def eval_status(self, data, formula):
        "`data` is a route plan object."
        labels_count = [0, 0, 0, 0]
        status = ['waiting', 'assigned', 'in_transit', 'dropped_off']
        for r in data.R:
            labels_count[status.index(r.current_status.name)] += 1

        if formula[0] == "D":
            return (labels_count[3] == int(formula[1]))
        elif formula[0] == "W":
            return (labels_count[0] == int(formula[1]))
        elif formula[0] == "A":
            return (labels_count[1] == int(formula[1]))
        elif formula[0] == "T":
            return (labels_count[2] == int(formula[1]))
        else:
            raise ValueError("Condition not in `D`, `W`, `A`, `T`.")
        

    def forall_next(self, node, formula, criterion):
        if not node.children:
            return False
        for child in node.children:
            if not child.evaluate(formula, criterion):
                return False
        return True
    

    def exists_next(self, node, formula, criterion):
        if not node.children:
            return False
        for child in node.children:
            if child.evaluate(formula, criterion):
                return True
        return False
    

    def find_all_paths(self, node):
        def dfs(node, current_path):
            if not node.children:
                all_paths.append(current_path.copy())
                return
            for child in node.children:
                current_path.append(child)
                dfs(child, current_path)
                current_path.pop()

        all_paths = []
        dfs(node, [node])
        return all_paths


    def single_path_future(self, path, formula):
        for nd in path:
            if nd.evaluate(formula):
                return True
        return False
    

    def single_path_global(self, path, formula):
        for nd in path:
            if not nd.evaluate(formula):
                return False
        return True
    

    def forall_future(self, node, formula):
        "AF"
        if node.evaluate(formula) and not node.children:
            return True
        
        all_paths_from_node = self.find_all_paths(node)
        for pth in all_paths_from_node:
            if not self.single_path_future(pth, formula):
                return False
            
        return True
    

    def exists_future(self, node, formula):
        "EF"
        if node.evaluate(formula):
            return True
    
        all_paths_from_node = self.find_all_paths(node)
        for pth in all_paths_from_node:
            if self.single_path_future(pth, formula):
                return True
            
        return False
    

    def forall_global(self, node, formula):
        "AG"
        if node.evaluate(formula) and not node.children:
            return True
        
        all_paths_from_node = self.find_all_paths(node)
        for pth in all_paths_from_node:
            if not self.single_path_global(pth, formula):
                return False
            
        return True
    

    def exists_global(self, node, formula):
        "EG"
        if node.evaluate(formula) and not node.children:
            return True
        
        all_paths_from_node = self.find_all_paths(node)
        for pth in all_paths_from_node:
            if self.single_path_global(pth, formula):
                return True
            
        return False
    



if __name__ == "__main__":
    ## example tree
    node1 = CTLNode(1, "A")
    node2 = CTLNode(2, "B")
    node3 = CTLNode(3, "C")
    node4 = CTLNode(4, "D")
    node5 = CTLNode(5, "E")
    node6 = CTLNode(6, "F")
    node1.add_child(node2)
    node1.add_child(node3)
    node2.add_child(node4)
    node2.add_child(node5)
    node3.add_child(node5)
    node3.add_child(node6)
    print_simple_tree(node1)

    # Evaluate CTL formulas on the tree
    print("Testing CTL model checker")

    # Node value
    print(node1.evaluate(("A")))  # Output: True

    # Node value
    print(node1.evaluate(("B")))  # Output: False

    # Exists next
    print(node1.evaluate(("EX", "B")))  # Output: True

    # Exists next
    print(node1.evaluate(("EX", "F")))  # Output: False

    # Always global
    print(node1.evaluate(("AG", "A")))  # Output: False

    # Exists future
    print(node1.evaluate(("EF", "F")))  # Output: True


