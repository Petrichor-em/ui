import json
import networkx as nx
from pyvis.network import Network

def load_graph_data(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    return kg_data

def print_nodes_and_edgs(G):
    for node, attributes in G.nodes(data=True):
        print(f"Node: {node}, Node_attribute: {attributes}")
    
    for edge in G.edges(data=True):
        print(f"Edge: {edge}")
    
def add_node_attribute(G, node_name, attribute_key, attribute_value):
    if node_name in G.nodes:
        attribute_value_str = str(attribute_value)
        G.nodes[node_name][attribute_key] = attribute_value_str
        return G
    else:
        print(f"ADD_ATTRIBUTES_ERROR: Node {node_name} is not existed!")
        return G

def parse_kg_from_json(kg_data, G):
    count=0
    for chunk_data in kg_data['results']:
        print(f"Chunk {count}:")
        nodes = chunk_data['nodes']
        attributes = chunk_data['attributes']
        relationships = chunk_data['relationships']
        # add nodes
        for key, value in nodes.items():
            for node_info in value:
                entity_name = node_info["entity_name"]
                entity_type = node_info["entity_type"]
                description = node_info["description"]
                G.add_node(entity_name, entity_type=entity_type, description=description)
        
        # add attributes
        for key, value in attributes.items():
            for attr_info in value:
                entity_name = key 
                attribute_key = attr_info["attribute_name"]
                # attribute_value = attr_info["attribute_value"]
                attribute_value = {key: value for key, value in attr_info.items() if key not in ["entity_name", "attribute_name"]}
                add_node_attribute(G, entity_name, attribute_key, attribute_value)
        
        # add relationships
        for key, value in relationships.items():
            for relation in value:
                src_id = relation["source_entity"]
                tgt_id = relation["target_entity"]
                edge_attributes = {key: value for key, value in relation.items() if key not in ["source_entity", "target_entity"]}
                if src_id not in G.nodes or tgt_id not in G.nodes:
                    print(f"ADD_RELATION_ERROR: Fail to add edge between {src_id} and {tgt_id}")
                    continue
                else:
                    G.add_edge(src_id, tgt_id, **edge_attributes)
        
        count+=1
    
    return G

