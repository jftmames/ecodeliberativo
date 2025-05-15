# trajectory_visualization.py

import graphviz

def build_deliberative_tree(steps):
    """
    Crea un diagrama tipo árbol/secuencia deliberativa con Graphviz.
    steps: lista de pasos (dictionaries)
    """
    dot = graphviz.Digraph(comment="Trayectoria Deliberativa", format="svg")
    prev_node = None
    for i, step in enumerate(steps):
        label = f"{i+1}. {step['question']}\n→ {step['answer']}"
        if step["context"].get("comentario"):
            label += f"\n({step['context']['comentario']})"
        node_id = f"n{i}"
        dot.node(node_id, label, shape='box', style='rounded,filled', color='#cde')
        if prev_node is not None:
            dot.edge(prev_node, node_id)
        prev_node = node_id
    return dot
