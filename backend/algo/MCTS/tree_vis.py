import colorsys
import math
import numpy as np
import networkx as nx
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams
from collections import defaultdict
import matplotlib.pyplot as plt
from pyvis.network import Network



def to_hex(decimal_val):
    deci_0 = format(int(decimal_val[0] * 255), '02X')
    deci_1 = format(int(decimal_val[1] * 255), '02X')
    deci_2 = format(int(decimal_val[2] * 255), '02X')
    return f'#{deci_0}{deci_1}{deci_2}'



def get_node_attr(tree_nodes, Q, N):
    "Get normalized score for each node."
    node_qual = defaultdict()
    for nd in Q.keys():
        if not N[nd] == 0:
            node_qual[nd] = Q[nd]/N[nd]
        else:
            node_qual[nd] = 0

    current_min = min(node_qual.values())
    current_max = max(node_qual.values())
    
    normalized_dict = defaultdict()
    for key, value in node_qual.items():
        normalized_value = ((value - current_min) / (current_max - current_min)) * 1
        normalized_dict[key] = normalized_value

    return node_qual, normalized_dict




def gradient_blue(num_colors = 10):
    start_color = (0, 60, 155)
    end_color = (160, 200, 255)
    gradient_colors = {}
    for i in range(num_colors+1):
        r = int(start_color[0] + (end_color[0] - start_color[0]) * i / (num_colors - 1))
        g = int(start_color[1] + (end_color[1] - start_color[1]) * i / (num_colors - 1))
        b = int(start_color[2] + (end_color[2] - start_color[2]) * i / (num_colors - 1))
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
        gradient_colors[i] = hex_color
    return gradient_colors




def gradient_rainbow(num_colors=10):
    values = np.linspace(0, 1, num_colors+1)
    colors = [plt.cm.viridis(val) for val in values]
    hex_colors = [mcolors.to_hex(color) for color in colors]
    interpolated_colors = {}
    for i in range(len(hex_colors)):
        interpolated_colors[i] = hex_colors[i]

    return interpolated_colors




def graph_tree_vis(tree, filename=None):
    "tree is a mcts tree's children dictionary."
    "Visualizing MCTS using pyvis."

    tree_Q = tree.Q
    tree_Q_breakdown = tree.breakdown_Q
    tree_N = tree.N
    tree = tree.children
    
    # assign each node for an id
    node_to_id = {}
    node_id = 0
    for k in tree.keys():
        node_to_id[k] = node_id
        node_id += 1

    for v in tree.values():
        for vv in v:
            if vv not in node_to_id:
                node_to_id[vv] = node_id
                node_id += 1

    node_qual, normalized_dict = get_node_attr(node_to_id.keys(), tree_Q, tree_N)

    # map node colors to a colormap 
    colormap = cm.get_cmap('viridis_r')
    node_colors = dict()
    for node in tree.keys():
        try:
            node_colors[node] = to_hex(colormap(normalized_dict[node]))
        except KeyError:
            node_colors[node] = r'#747474'
            normalized_dict[node] = 0


    for v in tree.values():
        for vv in v:
            if vv not in node_colors:
                try:
                    node_colors[vv] = to_hex(colormap(normalized_dict[vv]))
                except KeyError:
                    node_colors[vv] = r'#747474'
                    normalized_dict[vv] = 0


    def get_node_title(R, val=0, vis=0, uct=0, root=False):
        labels_count = [0, 0, 0, 0]
        status = ['waiting', 'assigned', 'in_transit', 'dropped_off']
        short = ['W', 'A', 'T', 'D']
        text = ""
        for r in R:
            labels_count[status.index(r.current_status.name)] += 1
        if root:
            text += "ROOT\n"
        
        text += "Total requests: {}\n".format(sum(labels_count))
        text += "Fulfilled requests: {}\n".format(labels_count[2]+labels_count[3])
        if uct == "--":
            text += "val: {:d}; vis: {}; uct: --".format(val, vis)
        else:
            text += "val: {:.2f}; vis: {}; uct: {:.2f}".format(val, vis, uct)
        ret = [st+str(num) for num, st in zip(labels_count, short)]
        return text
    

    def get_node_label(R):
        labels_count = [0, 0, 0, 0]
        status = ['waiting', 'assigned', 'in_transit', 'dropped_off']
        for r in R:
            labels_count[status.index(r.current_status.name)] += 1
        text = "There are currently {} requests waiting for vehicle assignment. \n".format(labels_count[0])
        text += "{} requests have been assigned but are waiting to be picked up. \n".format(labels_count[1])
        text += "{} requests are currently in transit. \n".format(labels_count[2])
        text += "{} requests have already been fulfilled.".format(labels_count[3])
        return text
    

    # add node attributes
    max_group = 0
    node_attributes = {}
    for k in tree.keys():
        if tree_N[k] > 0:
            log_N_vertex = math.log(tree_N[k])
            uct_score = tree_Q[k] / tree_N[k] + math.sqrt(log_N_vertex / tree_N[k])
        else:
            uct_score = '--'
        if len(k.R) > 0:
            node_attributes[k] = get_node_title(k.R, tree_Q[k], tree_N[k], uct_score)
        else:
            node_attributes[k] = get_node_title([k.r_t], tree_Q[k], tree_N[k], uct_score, root=True)
        if len(k.R) > max_group:
            max_group = len(k.R)
    
    for v in tree.values():
        for vv in v:
            if vv not in node_attributes:
                if tree_N[vv] > 0:
                    log_N_vertex = math.log(tree_N[vv])
                    uct_score = tree_Q[vv] / tree_N[vv] + math.sqrt(log_N_vertex / tree_N[vv])
                else:
                    uct_score = '--'
                if len(vv.R) > 0:
                    node_attributes[vv] = get_node_title(vv.R, tree_Q[vv], tree_N[vv], uct_score)
                else:
                    node_attributes[vv] = get_node_title([vv.r_t], tree_Q[vv], tree_N[vv], uct_score)
                if len(vv.R) > max_group:
                    max_group = len(vv.R)


    # add node titles
    node_titles = {}
    for k in tree.keys():
        if len(k.R) > 0:
            node_titles[k] = get_node_label(k.R)
        else:
            node_titles[k] = get_node_label([k.r_t])
        if len(k.R) > max_group:
            max_group = len(k.R)
    for v in tree.values():
        for vv in v:
            if vv not in node_titles:
                if len(vv.R) > 0:
                    node_titles[vv] = get_node_label(vv.R)
                else:
                    node_titles[vv] = get_node_label([vv.r_t])
                if len(vv.R) > max_group:
                    max_group = len(vv.R)


    # create pyvis figure object.
    nt = Network('1500px', '1500px', directed=True)
    group_color = gradient_rainbow(num_colors=max_group)
    

    # add nodes.
    for k in tree.keys():
        nt.add_node(node_to_id[k], label=node_attributes[k], color=group_color[len(k.R)], 
                    borderWidth=3, borderWidthSelected=4, size=30, labelHighlightBold=True,
                    title = node_titles[k], mass=2.5)
    for v in tree.values():
        for vv in v:
            try:
                nt.get_node(node_to_id[vv])
            except:
                nt.add_node(node_to_id[vv], label=node_attributes[vv], color=group_color[len(vv.R)],
                            borderWidth=3, borderWidthSelected=4, size=30, labelHighlightBold=True,
                            title = node_titles[vv], mass=2.5)
    
    def get_edge_label(R):
        text = "Request assigned to vehicle {}".format(R[-1].assigned_vehicle)
        return text 

    # add edges.
    for k,v in tree.items():
        for vv in v:
            nt.add_edge(node_to_id[k], node_to_id[vv], width=3, label=get_edge_label(vv.R))
            

    # create html graph.
    nt.toggle_physics(True)

    if filename:
        nt.save_graph(filename)
    else:
        nt.show_buttons()
        nt.show('mcts-pyvis.html', notebook=False)





def graph_tree_vis_plt(tree):
    "tree is a mcts tree's children dictionary."

    # define a graph 
    G = nx.DiGraph(tree)

    # define custom node colors
    def get_node_color(node, min_value=1, max_value=6):
        int_color_val = int(len(node.R))
        return (int_color_val-min_value) / (max_value-min_value)
    
    # map node colors to a colormap 
    colormap = cm.get_cmap('viridis')
    node_colors = {node: colormap(get_node_color(node)) for node in G.nodes()}

    # add node attributes
    node_attributes = {}
    for k in tree.keys():
        if len(k.R) > 0:
            node_attributes[k] = str(len(k.R))+":"+k.R[-1].current_status.name
        else:
            node_attributes[k] = str(len(k.R))+":"+k.r_t.current_status.name
    nx.set_node_attributes(G, values=node_attributes, name='trip_status')

    # add the list of paths 
    nodes_list = tree.keys()
    nx.add_path(G, nodes_list)

    # get node labels
    labels = nx.get_node_attributes(G, 'trip_status') 
    
    rcParams['figure.figsize'] = 14, 10
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos,
            node_color=list(node_colors.values()), 
            node_size=250,
            with_labels=True, 
            labels=labels,
            arrows=True)
    
    plt.show()