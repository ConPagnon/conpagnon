import networkx as nx
from conpagnon.utils.folders_and_files_management import load_object
import numpy as np
from conpagnon.data_handling import atlas

"""
Export networkx graph to other for format for network visualisation.

"""


# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=labels_text_file,
    normalize_colors=False)

# Load mean connectivity matrix
metric = 'correlation'
group = 'patients'
mean_matrix = load_object('/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                          'mean_connectivity_matrices_patients_controls.pkl')[group][metric]

# Build graph
threshold = 0.5
mean_matrix[np.abs(mean_matrix) < threshold] = 0

G = nx.Graph(incoming_graph_data=mean_matrix)

# GEXF format for gephi
#for item in G.nodes(data=True):
#    G.node[item[0]]['viz'] = {'color': {'r': int(labels_colors[item[0]][0]), 'g': int(labels_colors[item[0]][1]),
#                                        'b': int(labels_colors[item[0]][2]), 'a': 0}}
#    G.node[item[0]]['label'] = labels_regions[item[0]]

# Write to the GEXF format for gephi
#nx.write_gexf(G, "/media/db242421/db242421_data/Bac_a_sable/test_gephi/test_patients.gexf", version="1.2draft")

# GML format for cytoscape
nodes_name_dict = {i: labels_regions[i] for i in range(len(labels_regions))}
nx.set_node_attributes(G, labels_regions, 'labels')

# GML format for cytoscape
nx.write_gml(G, "/media/db242421/db242421_data/Bac_a_sable/test_gephi/test_patients.gml")


import networkx as nx

G = nx.Graph()
G.add_edge(0, 1, weight=0.1, label='edge', graphics={
    'width': 1.0, 'fill': '"#0000ff"', 'type': '"line"', 'Line': [],
    'source_arrow': 0, 'target_arrow': 0})
nx.set_node_attributes(G, 'graphics', {
    0: {'x': -85.0, 'y': -97.0, 'w': 20.0, 'h': 20.0,
        'type': '"ellipse"', 'fill': '"#889999"', 'outline': '"#666666"',
        'outline_width': 1.0},
    1: {'x': -16.0, 'y': -1.0, 'w': 40.0, 'h': 40.0,
        'type': '"ellipse"', 'fill': '"#ff9999"', 'outline': '"#666666"',
        'outline_width': 1.0}
    })
nx.set_node_attributes(G, 'label', {0: "0", 1: "1"})
nx.write_gml(G, '/media/db242421/db242421_data/Bac_a_sable/test_gephi/network.gml')