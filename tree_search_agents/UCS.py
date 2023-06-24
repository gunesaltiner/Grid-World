"""
    Name: Güneş
    Surname: Altıner
    Student ID: 020893
"""

from tree_search_agents.TreeSearchAgent import *
from tree_search_agents.PriorityQueue import PriorityQueue
import time


class UCSAgent(TreeSearchAgent):
    def run(self, env: Environment) -> (List[int], float, list):
        """
            You should implement this method for Uniform Cost Search algorithm.

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
        queue.enqueue([current_node, [], top_reward, is_done], top_reward)

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
                        queue.enqueue(obj, top_reward_neighbour)

                    env.set_current_state(state)

        return [], 0, []
    @property
    def name(self) -> str:
        return "UCS"
