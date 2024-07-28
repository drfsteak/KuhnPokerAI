from typing import List, Dict
import random
import numpy as np
import sys

Actions = ['B', 'C']  # bet/call vs check/fold
CARDNAMES = ['9', 'T', 'J', 'Q', 'K']

class InformationSet():
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=len(Actions))
        self.strategy_sum = np.zeros(shape=len(Actions))
        self.num_actions = len(Actions)

    def normalize(self, strategy: np.array) -> np.array:
        """Normalize a strategy. If there are no positive regrets,
        use a uniform random strategy"""
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        """Return regret-matching strategy"""
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())

    def get_average_strategy_with_threshold(self, threshold: float) -> np.array:
        avg_strat = self.get_average_strategy()
        avg_strat[avg_strat < threshold] = 0
        return self.normalize(avg_strat)


class KuhnPoker():
    @staticmethod
    def all_opponents_folded(history: str, num_players: int):
        return len(history) >= num_players and history.endswith('C' * (num_players - 1))

    @staticmethod
    def get_payoff(cards: List[str], history: str, num_players: int) -> List[int]:
        """
        Returns the payoff for all terminal game nodes.
        """
        player = len(history) % num_players
        player_cards = cards[:num_players]
        num_opponents = num_players - 1
        if history == 'C' * num_players:
            payouts = [-1] * num_players
            payouts[np.argmax(player_cards)] = num_opponents
            return payouts
        elif KuhnPoker.all_opponents_folded(history, num_players):
            payouts = [-1] * num_players
            payouts[player] = num_opponents
        else:
            payouts = [-1] * num_players
            active_cards = []
            active_indices = []
            for (ix, x) in enumerate(player_cards):
                if 'B' in history[ix::num_players]:
                    payouts[ix] = -2
                    active_cards.append(x)
                    active_indices.append(ix)
            payouts[active_indices[np.argmax(active_cards)]] = len(active_cards) - 1 + num_opponents
        return payouts

    @staticmethod
    def is_terminal(history: str, num_players: int) -> bool:
        """
        Checks if a given history corresponds to a terminal state
        """
        all_raise = history.endswith('B' * num_players)
        all_acted_after_raise = (history.find('B') > -1) and (len(history) - history.find('B') == num_players)
        all_but_1_player_folds = KuhnPoker.all_opponents_folded(history, num_players)
        return all_raise or all_acted_after_raise or all_but_1_player_folds
    
    # in progress 08/02/2024
