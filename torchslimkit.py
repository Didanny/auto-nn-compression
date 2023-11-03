import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.onnx import TrainingMode

import onnx

from typing import Optional, Sequence, Union, TypedDict
from typing_extensions import Literal, TypeAlias

# Node class definition
class Node:
    def __init__(self, name, node_type, inputs, outputs):
        self.name = name  # str
        self.node_type = node_type  # str
        self.inputs = inputs  # List[str]
        self.outputs = outputs  # List[str]
        self.parent = []  # parent node names
        self.child_nodes = []  # child node names

    def __repr__(self):
        return f"node_type={self.node_type}, parent={', '.join(self.parent)}, child_nodes={', '.join(self.child_nodes)}"

def name2attr(name):
    new_name = ''
    for s in name.split('.'):
        if s.isdigit():
            new_name += f'[{s}]'
        else:
            new_name += f'.{s}'
    return new_name

def get_filter_indices(module: nn.Module, pruned: Optional[bool] = True):
    check = (lambda x: x==0) if pruned else (lambda x: x!=0)    
    return (check(torch.norm(module.weight.data, 1, (1, 2, 3)))).nonzero().view(-1)   

def get_kernel_indices(module: nn.Module, pruned: Optional[bool] = True):
    check = (lambda x: x==0) if pruned else (lambda x: x!=0) 
    return (check(torch.norm(module.weight.data, 1, (0, 2, 3)))).nonzero().view(-1) 

def generate_onnx_graph(model: nn.Module, input_size: Optional[tuple] = (1, 3, 600, 600), preprocess: Optional[callable] = None):
    # Create dummy input for model tracing
    dummy_input = torch.rand(input_size)
    if preprocess != None:
        dummy_input = preprocess(dummy_input)
        
    # Trace and export the model
    # TODO: Uncomment this line
    # torch.onnx.export(model, dummy_input, 'out.onnx', training=TrainingMode.TRAINING)
    
    # Load the model
    onnx_model = onnx.load('out.onnx')
    
    # Rename onnx layers with PyTorch naming scheme
    for i in range(len(onnx_model.graph.node)):
        if len(onnx_model.graph.node[i].input) > 1:
            onnx_model.graph.node[i].name = onnx_model.graph.node[i].input[1].split('.weight')[0]
    
    # TODO: Remove this line (Used for Netron debugging)        
    onnx.save(onnx_model, 'out.onnx')
            
    # Return the graph
    return onnx_model.graph

def generate_graph(onnx_graph):
    node_dict = {}

    # Create all the nodes from onnx file
    for node in onnx_graph.node:
        node_name = node.name
        node_type = node.op_type
        node_inputs = list(node.input)
        node_outputs = list(node.output)

        node_dict[node_name] = Node(node_name, node_type, node_inputs, node_outputs)  # create Node instance

    # Find dependencies each other
    for node in onnx_graph.node:
        current_node = node_dict[node.name]
        # Check one's output with other's input
        for output_name in node.output:
            for another_node in onnx_graph.node:
                if output_name in another_node.input:
                    # If matched
                    if another_node.name in node_dict:
                        # append to the list
                        current_node.child_nodes.append(another_node.name)  # for child nodes
                        node_dict[another_node.name].parent.append(current_node.name)  # for parent nodes

    return node_dict  # key : node_name / value : Node instance

def _get_all_parents(node: str, graph: dict[str, Node], out: list[str]):
    node = graph[node]
    if node.node_type == 'Conv':
        out.append(node.name)
    else:
        for parent in node.parent:
            _get_all_parents(parent, graph, out)
            
def get_all_parents(node: str, graph: dict[str, Node]):
    out = []
    node = graph[node]
    
    for parent in node.parent:
        _get_all_parents(parent, graph, out)
        
    return out

def _get_all_children(node: str, graph: dict[str, Node], out: list[str]):
    node = graph[node]
    if node.node_type == 'Gemm':
        out.append(node.name)
    else:
        for child in node.child_nodes:
            _get_all_children(child, graph, out)
            
def get_all_children(node: str, graph: dict[str, Node]):
    out = []
    node = graph[node]
    
    for child in node.child_nodes:
        _get_all_children(child, graph, out)
        
    return out

def embed_filter_indices(model: nn.Module):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        
        filter_indices = get_filter_indices(mod, pruned=False)
        mod.filter_indices = filter_indices    
        
def embed_kernel_indices(model: nn.Module, graph: dict[str, Node]):
    for name, mod in model.named_modules():
        # Get the parent layer
        parent = get_all_parents(name, graph)
        if len(parent) < 1:
            continue
        parent = parent[0]
        parent = exec(f'modl{name2attr(parent)}')
        
        # Embed the filter_indices of the paren as kernel_indices in the child
        mod.kernel_indices = parent.filter_indices
    
def aggregate_sibling_groups(graph: dict[str, Node]):
    sibling_groups = []
    all_sibling_groups = []

    for node in graph.keys():
        node = graph[node]
        if node.node_type == 'Add':
            group = get_all_parents(node.name, graph)
            all_sibling_groups.append(group)

    # First and last group edge cases
    if len(all_sibling_groups[0]) >= len(all_sibling_groups[1]):
        sibling_groups.append(all_sibling_groups[0])
    if len(all_sibling_groups[-1]) >= len(all_sibling_groups[-2]):
        sibling_groups.append(all_sibling_groups[-1])

    # Rest of the groups
    for i in range(1, len(all_sibling_groups)-1):
        if len(all_sibling_groups[i]) >= len(all_sibling_groups[i-1]) and len(all_sibling_groups[i]) >= len(all_sibling_groups[i+1]):
            sibling_groups.append(all_sibling_groups[i])

    return sibling_groups

def _update_filter_indices(model: nn.Module, group: list[str]):
    filter_idx_list = []
    for mod in group:
        mod = eval(f'model{name2attr(mod)}')
        filter_idx_list.append(mod.filter_indices)
        
    filter_indices = torch.cat(filter_idx_list)
    filter_indices = torch.unique(filter_indices)
    filter_indices, _ = torch.sort(filter_indices)

    for mod in group:
        mod = eval(f'model{name2attr(mod)}')
        exec(f'mod.filter_indices = filter_indices')

def update_filter_indices(model: nn.Module, groups: list[list[str]]):
    for group in groups:
        _update_filter_indices(model, group)
        
def embed_final_kernel_indices(model: nn.Module, graph: dict[str, Node]):
    # Get any conv layer
    pass
    
    # Get all the fully connected layers, who are immediate children of conv layers
    
    

def slimify(model: nn.Module, **kwargs):
    # Generate onnx graph
    onnx_graph = generate_onnx_graph(model, **kwargs)
    
    # Generate dependency graph
    graph = generate_graph(onnx_graph=onnx_graph)
    
    # Embed un-pruned filter indices into model
    embed_filter_indices(model)
    
    # Get and aggregate the sibling dependencies
    sibling_groups = aggregate_sibling_groups(graph)
    
    # Update the embedded filter indices based on the sibling groups
    update_filter_indices(model, sibling_groups)
    
    # Embed un-pruned kernel indices into model
    embed_kernel_indices(model, graph)
    
    # Enbed kernel indices into the first fully connected layer
    embed_final_kernel_indices(model, graph)
    
    # Reconstruct the network
        