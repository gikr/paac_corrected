from ..env_T_2 import make_game, T_lab_observation, T_lab_actions

from ..environment import BaseEnvironment

import numpy as np
import operator


class TLabyrinthEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.randomness = True
        self.reward_location = None

        self.legal_actions = T_lab_actions()
        self.noop = 'pass'
        self.id = actor_id

        self.game = make_game(self.randomness, self.reward_location, False)
        obs_t, r_t, discount_t = self.game.its_showtime()
        obs_t, info = T_lab_observation(obs_t)
        self.observation_shape = obs_t.shape

        self.enlarge = False
        self.rewards = []

    def reset(self):
        """Starts a new episode and returns its initial state"""

        self.game = make_game(self.randomness, self.reward_location, False)
        obs_t, r_t, discount_t = self.game.its_showtime()
        obs, info = T_lab_observation(obs_t)

        return obs, None

    def next(self, action):

        """
        Performs the given action.
        Returns the next state, reward, and terminal signal
        """
        act = [i for i, x in enumerate(action) if x]

        if not self.game.game_over:
            obs, reward, discount = self.game.play(act[0])
        termination = 1 - discount

        observation, info = T_lab_observation(obs)
        ls = []
        # if len(str(self.id)) == 1:
        data = [[], []]
        if self.id != data[0] and reward != -1:
            data[0].append(self.id)
            data[1].append(reward)
            # print(data)
            # print("actor, reward, termination", self.id, reward, termination)
            # if data[0]==[4]: #write to file
            # print(data)
        # print(reward, termination)

        if termination == True:
            self.rewards.append(reward)
            if len(self.rewards) == 51:
                self.rewards = self.rewards[1:]
                if len(np.where(np.array(self.rewards) > 0)) >= 45:
                    print(self.rewards)
                    # self.enlarge = True

        return observation, reward, termination, None

    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def on_new_frame(self, frame):
        pass

    def close(self):
        pass
