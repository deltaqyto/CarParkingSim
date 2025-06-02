class GenericModule:
    def __init__(self):
        pass

    def reset(self, mode, state):
        pass

    def get_digest(self):
        raise NotImplementedError("Tried to get digest on generic class")

    def get_unified_state(self):
        return {}

    def render(self, screen, transform_matrix):
        pass


class GenericReward(GenericModule):
    def __init__(self):
        super().__init__()

    def get_reward(self, state):
        return 0

class GenericEnvironment(GenericModule):
    def __init__(self):
        super().__init__()

    def step(self, state):
        pass


class GenericStop(GenericModule):
    def __init__(self):
        super().__init__()

    def check_stop(self, state):
        return True, 'Called On Generic Module'


class GenericObservation(GenericModule):
    def __init__(self):
        super().__init__()

    def get_observation(self, state, observation):
        # Observation input contains info from previous modules if multiple modules are strung together
        raise NotImplementedError("Tried to get observation from generic module")
