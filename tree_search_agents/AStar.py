"""
    Name: Güneş
    Surname: Altıner
    Student ID: 020893
"""
import numpy as np

from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class AStarAgent(TreeSearchAgent):
    def run(self, env: Environment) -> (List[int], float, list):
        """
            You should implement this method for A* algorithm.

            DO NOT CHANGE the name, parameters and output of the method.
        :param env: Environment
        :return: List of actions and total score
        """

        visited = set()
        queue = PriorityQueue()

        current_node = env.to_state(env.starting_position)
        top_reward = 0
        is_done = False

        goals = env.get_goals()

        queue.enqueue([current_node, [], top_reward, is_done], 0)

        while not queue.is_empty():
            dequeued = queue.dequeue()
            state = dequeued[0]
            action_list = dequeued[1]
            top_reward = dequeued[2]
            env.set_current_state(dequeued[0])

            if state not in visited:
                visited.add(state)

                if state in goals:
                    return action_list, top_reward, visited

                for i in range(4):
                    action_list = dequeued[1]
                    top_reward_current = top_reward
                    new_state, new_reward, is_done = env.move(i)
                    neighbor_action = action_list.copy()

                    if new_state not in visited:
                        top_reward_neighbour = top_reward_current + new_reward
                        neighbor_action.append(i)

                        obj = [new_state, neighbor_action, top_reward_neighbour, is_done]
                        new_heuristic_value = self.get_heuristic(env, new_state, goal=goals)
                        new_f_value = top_reward_neighbour - new_heuristic_value
                        queue.enqueue(obj, new_f_value)

                    env.set_current_state(state)

        return [], 0, []

    def get_heuristic(self, env: Environment, state: int, **kwargs) -> float:
        """
            You should implement your heuristic calculation for A*

            DO NOT CHANGE the name, parameters and output of the method.

            Note that you can use kwargs to get more parameters :)
        :param env: Environment object
        :param state: Current state
        :param kwargs: More parameters
        :return: Heuristic score
        """

        goals = list(kwargs.values())
        state_position = env.to_position(state)

        heuristic_list = []

        for i in goals[0]:
            goal_position = env.to_position(i)
            heuristic_value = abs(goal_position[0]-state_position[0]) + abs(goal_position[1]-state_position[1])
            heuristic_list.append(heuristic_value)

        min_heuristic_value = np.min(heuristic_list)

        return min_heuristic_value

    @property
    def name(self) -> str:
        return "AStar"
