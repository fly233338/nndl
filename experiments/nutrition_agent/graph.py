def build_graph():
    from langgraph.graph import END, START, StateGraph
    from .nodes import nutrition_node, validate_node
    graph = StateGraph(dict)
    graph.add_node("validate", validate_node)
    graph.add_node("nutrition", nutrition_node)
    graph.add_edge(START, "validate")
    graph.add_edge("validate", "nutrition")
    graph.add_edge("nutrition", END)
    return graph.compile()
