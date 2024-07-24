from enum import Enum
from random import choices
from typing import List
import numpy as np

class Action(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

cumulative_regrets = np.zeros(shape=len(Action), dtype='int')
strategy_sum = np.zeros(shape=len(Action))

opp_strategy = [0.8, 0.1, 0.1]

def get_payoff(action_1: Action, action_2: Action) -> int:
    result = (action_1.value - action_2.value) % 3
    if result == 2:
        return -1
    else:
        return result

def get_strategy(cumulative_regrets) -> np.array:
    pos_cumulative_regrets = np.maximum(0, cumulative_regrets)
    if sum(pos_cumulative_regrets) > 0:
        return pos_cumulative_regrets / sum(pos_cumulative_regrets)
    else:
        return np.full(shape=len(Action), fill_value=1/len(Action))

def get_regrets(payoff: int, action_2: Action) -> np.array:
    return np.array([get_payoff(a, action_2) - payoff for a in Action])

num_iterations = 10000

for _ in range(num_iterations):
    strategy = get_strategy(cumulative_regrets)
    strategy_sum += strategy

    player_action = choices(list(Action), weights=strategy)[0]
    opp_action = choices(list(Action), weights=opp_strategy)[0]

    player_payoff = get_payoff(player_action, opp_action)
    regrets = get_regrets(player_payoff, opp_action)
    
    cumulative_regrets += regrets

optimal_strategy = strategy_sum / num_iterations

np.set_printoptions(formatter={'float': '{:0.6f}'.format})
print(optimal_strategy)