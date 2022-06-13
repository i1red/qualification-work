import random


class RandomWalker:
    def __init__(self, adj_list: list[list[int]], p: float, q: float) -> None:
        self.adj_list = list(adj_list)
        self.adj_set_list = [set(neighbor_nodes) for neighbor_nodes in adj_list]
        self.nodes = list(range(len(adj_list)))
        self.p = p
        self.q = q

    def generate_walks(self, num_walks_per_node: int,
                       walk_length: int, print_progress: bool = False) -> list[list[int]]:
        walks = []

        for walks_iteration in range(1, num_walks_per_node + 1):
            nodes = list(self.nodes)
            random.shuffle(nodes)

            for walk_no, first_node in enumerate(nodes, 1):
                if print_progress:
                    print(f'Walks Iteration #{walks_iteration}/{num_walks_per_node}. '
                          f'Walk #{walk_no}/{len(nodes)}\r', end='')

                walk = self.perform_walk(first_node, walk_length)
                walks.append(walk)

            if print_progress:
                print()

        return walks

    def perform_walk(self, first_node: int, walk_length: int) -> list[int]:
        walk = [first_node]

        for _ in range(walk_length - 1):
            current_node = walk[-1]
            previous_node = walk[-2] if len(walk) > 1 else None
            next_node = self.choose_next_node(previous_node, current_node)
            walk.append(next_node)

        return walk

    def choose_next_node(self, previous_node: int, current_node: int) -> int:
        neighbor_nodes = self.adj_list[current_node]

        step_probabilities = [self.calculate_step_probability(neighbor_node, previous_node)
                              for neighbor_node in neighbor_nodes]

        next_node, = random.choices(neighbor_nodes, weights=step_probabilities, k=1)
        return next_node

    def calculate_step_probability(self, neighbor_node: int, previous_node: int) -> float:
        if neighbor_node == previous_node:
            return 1. / self.p

        if previous_node in self.adj_set_list[neighbor_node]:
            return 1.

        return 1. / self.q