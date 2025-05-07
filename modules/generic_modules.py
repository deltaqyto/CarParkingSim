class GenericModule:
    def __init__(self):
        pass

    def reset(self, state):
        pass

    def get_digest(self):
        raise NotImplementedError("Tried to get digest on generic class")

    def get_unified_state(self):
        return {}


class GenericReward(GenericModule):
    def __init__(self):
        super().__init__()

    def get_reward(self, state):
        return 0

class GenericEnvironment(GenericModule):
    def __init__(self):
        super().__init__()

    def reset(self, collision_system, state):
        pass

    def render(self, screen, transform_matrix):
        pass


class GenericStop(GenericModule):
    def __init__(self):
        super().__init__()

    def check_stop(self, state):
        return True, 'Called On Generic Module'

    def render(self, screen, transform):
        pass