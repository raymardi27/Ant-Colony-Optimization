graph = []
pheromones = []
initial_pheromone_level = 0

def initialize_pheromones():
    for edge in graph:
        pheromones[edge] = initial_pheromone_level
ants = []
def build_solution_for_ant(ant):
    pass

def construct_solutions():
    solutions = []
    for ant in ants:
        solution_k = build_solution_for_ant(ant)
        solutions.append(solution_k)
    return solutions

start_node = 0
end_node = 0
choose_next_node = ()

def build_solution_for_ant(ant):
    solution = []
    current_node = start_node
    while current_node != end_node:
        next_node = choose_next_node(current_node)
        solution.append(next_node)
        current_node = next_node
    return solution

def get_neighbors(node):
    pass
alpha,heuristic_info,beta = 0, 0,0

def choose_next_node(current_node):
    neighbors = get_neighbors(current_node)
    weights = []
    for n in neighbors:
        weights[n] = (pheromones[(current_node,n)]^alpha) * (heuristic_info[(current_node,n)]^beta)
    sum_weights = sum(weights)
    r = random(0, sum_weights)
    cumulative = 0
    for n in neighbors:
        cumulative += weights[n]
        if r <= cumulative:
            return n


cost = ()

def find_best_solution(solutions):
    best = solutions[0]
    for s in solutions:
        if cost(s) < cost(best):
            best = s
    return best

evaporation_rate = 0

def evaporate_pheromones():
    for edge in graph:
        pheromones[edge] = (1 - evaporation_rate) * pheromones[edge]

pheromone_deposit = 0

def deposit_pheromones(best_solution):
    for edge in best_solution:
        pheromones[edge] += pheromone_deposit / cost(best_solution)

