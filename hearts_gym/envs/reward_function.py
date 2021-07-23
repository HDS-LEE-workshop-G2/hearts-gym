"""
The reward function an agent optimizes to win at Hearts.
"""

import numpy as np

from hearts_gym.utils.typing import Reward
from .hearts_env import HeartsEnv
from ..envs.card_deck import Card


class RewardFunction:
    """
    The reward function an agent optimizes to win at Hearts.

    Calling this returns the reward.
    """

    def __init__(self, env: HeartsEnv):
        self.env = env
        self.game = env.game

    def __call__(self, *args, **kwargs) -> Reward:
        return self.compute_reward(*args, **kwargs)

    def compute_reward(
            self,
            player_index: int,
            prev_active_player_index: int,
            trick_is_over: bool,
    ) -> Reward:
        """Return the reward for the player with the given index.

        It is important to keep in mind that most of the time, the
        arguments are unrelated to the player getting their reward. This
        is because agents receive their reward only when it is their
        next turn, not right after their turn. Due to this peculiarity,
        it is encouraged to use `self.game.prev_played_cards`,
        `self.game.prev_was_illegals`, and others.

        Args:
            player_index (int): Index of the player to return the reward
                for. This is most of the time _not_ the player that took
                the action (which is given by `prev_active_player_index`).
            prev_active_player_index (int): Index of the previously
                active player that took the action. In other words, the
                active player index before the action was taken.
            trick_is_over (bool): Whether the action ended the trick.

        Returns:
            Reward: Reward for the player with the given index.
        """
        if self.game.prev_was_illegals[player_index]:
            return -self.game.max_penalty * self.game.max_num_cards_on_hand

        card = self.game.prev_played_cards[player_index]

        if card is None:
            # The agent did not take a turn until now; no information
            # to provide.
            return 0
        
        # assign a reward to a card on your hand
        card_value = 0
        if card.suit == Card.SUIT_SPADE and card.rank == self.game.RANK_QUEEN:
            card_value = self.game.get_penalty(card)
        elif card.suit == Card.SUIT_HEART:
            card_value = self.game.get_penalty(card) * 0.5 * (card.rank / 13.) ** 2 * 13
        else:
            card_value = 0.25 * (card.rank / 13.) ** 2 * 13
            

        if trick_is_over and self.game.has_shot_the_moon(player_index):
            return self.game.max_penalty * self.game.max_num_cards_on_hand

        if self.game.prev_trick_penalty is not None:
            if self.game.prev_trick_winner_index == player_index and self.game.prev_leading_player_index == player_index:
                return -self.game.prev_trick_penalty - max(self.game.get_penalty(card),0.5)*2
            elif self.game.prev_trick_winner_index == player_index and self.game.prev_leading_player_index != player_index:
                return -self.game.prev_trick_penalty - max(self.game.get_penalty(card),0.5)
            elif self.game.prev_trick_winner_index != player_index and self.game.prev_leading_player_index == player_index:
                return self.game.prev_trick_penalty + max(self.game.get_penalty(card),0.5)*2
            elif self.game.prev_trick_winner_index != player_index and self.game.prev_leading_player_index != player_index:
                return self.game.prev_trick_penalty + max(self.game.get_penalty(card),0.5)
        else:
            return card_value / 13
