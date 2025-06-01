from modules.generic_modules import GenericReward


class GoalEndReward(GenericReward):
    def __init__(self, reward=20):
        super().__init__()
        self.reward = reward

    def get_digest(self):
        return f'GoalEndReward(reward={self.reward})'

    def get_reward(self, state):
        return 'Goal End', self.reward if 'Goal Hit' in state['stop_reasons'] else 0


class TimePenalty(GenericReward):
    def __init__(self, reward=-0.01):
        super().__init__()
        self.reward = reward

    def get_digest(self):
        return f'TimePenalty(reward={self.reward})'

    def get_reward(self, state):
        return 'Time Penalty', self.reward


class CollisionPenalty(GenericReward):
    def __init__(self, reward=-10):
        super().__init__()
        self.reward = reward

    def get_digest(self):
        return f'CollisionPenalty(reward={self.reward})'

    def get_reward(self, state):
        return 'Collision Penalty', self.reward if state['collisions'] else 0


class DistanceReward(GenericReward):
    def __init__(self, reward_factor=-1/6):
        super().__init__()
        self.reward_factor = reward_factor

    def get_digest(self):
        return f'DistanceReward(reward_factor={self.reward_factor})'

    def get_reward(self, state):
        goal_distance = state['closest_goal']['distance']
        return 'DistanceReward', self.reward_factor * goal_distance if state['stop_reasons'] else 0
