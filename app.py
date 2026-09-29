import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {},
}


# Heuristic: Euclidean Distance
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Path reconstruction
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# Greedy Best-First Search (GBFS)
def gbfs(start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    came_from = {start: None}
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            # Compute actual cumulative path cost
            cost = sum(
                hospital_graph[path[i]][path[i + 1]]
                for i in range(len(path) - 1)
            )
            return path, cost

        for neighbor in hospital_graph[current]:
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(
                    frontier, (heuristic(neighbor, goal), neighbor)
                )

    return None, float("inf")


# A* Search Algorithm
def a_star(start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    came_from = {start: None}
    g_cost = {start: 0.0}
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[goal]

        for neighbor, weight in hospital_graph[current].items():
            tentative_g = g_cost[current] + weight
            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f_cost = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_cost, neighbor))

    return None, float("inf")


##########################################
# Streamlit GUI Code

# Set Page Config
st.set_page_config(
    page_title="Emergency Supply Robot Pathfinding",
    layout="wide",
)

# Write title and description for the app
st.title("Hospital Emergency Supply Robot Pathfinding Visualizer")
st.markdown(
    "Select an initial and goal location, then choose a search strategy (GBFS or A*) "
    "to compute and visualize the trajectory through the hospital network."
)

# Define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# Create selectboxes for start, goal, and algorithm selection
start = st.selectbox(
    "Select Initial Node",
    options=nodes,
    index=nodes.index("Pharmacy"),
)

goal = st.selectbox(
    "Select Goal Node",
    options=nodes,
    index=nodes.index("Emergency_Ward"),
)

algorithm = st.selectbox(
    "Select Search Algorithm",
    options=["GBFS", "A* Search"],
)

if st.button("Run Search"):
    if algorithm == "GBFS":
        path, cost = gbfs(start, goal)
    else:
        path, cost = a_star(start, goal)

    if path is None:
        st.error(f"No path found between {start} and {goal}.")
    else:
        # Display result
        st.subheader("Search Result")

        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Solution Path:** {' → '.join(path)}")
        st.write(f"**Total Path Cost:** {cost:.2f}")

        # Visualize NetworkX graph
        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = locations

        fig, ax = plt.subplots(figsize=(10, 6))

        # Base graph layout
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color="skyblue", node_size=2200
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=9, font_weight="bold"
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            min_source_margin=20,
            min_target_margin=20,
        )

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        # Highlight solution path
        if len(path) > 1:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(
                G, pos, nodelist=path, node_color="orange", node_size=2200, ax=ax
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                edge_color="crimson",
                width=3,
                arrows=True,
                arrowsize=22,
                ax=ax,
            )

        ax.set_title(f"{algorithm} Solution Path", fontsize=14, fontweight="bold")
        ax.axis("off")

        st.pyplot(fig)