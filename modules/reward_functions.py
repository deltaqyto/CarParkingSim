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


class SmoothCollisionPenalty(GenericReward):
    def __init__(self, reward=-10, car_penalty_multiplier=2.0):
        super().__init__()
        self.reward = reward
        self.car_penalty_multiplier = car_penalty_multiplier

    def get_digest(self):
        return f'SmoothCollisionPenalty(reward={self.reward}, car_multiplier={self.car_penalty_multiplier})'

    def get_reward(self, state):
        if not state['collisions']:
            return 'Smooth Collision Penalty', 0
        
        # Enhanced penalty for parking environments (hitting cars vs walls)
        penalty = self.reward * self.car_penalty_multiplier
        return 'Smooth Collision Penalty', penalty


class SmoothDistanceReward(GenericReward):
    def __init__(self, reward_factor=-1/6, continuous=False, continuous_scale=0.05):
        super().__init__()
        self.reward_factor = reward_factor
        self.continuous = continuous
        self.continuous_scale = continuous_scale
        self.last_distance = None

    def get_digest(self):
        return f'SmoothDistanceReward(factor={self.reward_factor}, continuous={self.continuous})'

    def get_reward(self, state):
        # Get YOLO goal distance
        closest_goal_data = state.get('closest_goal', {})
        goal_distance = closest_goal_data.get('distance', float('inf'))
        
        # If no YOLO goals detected yet, return small exploration penalty
        if goal_distance == float('inf'):
            return 'SmoothDistanceReward', -0.001 if self.continuous else 0
        
        if self.continuous:
            # Continuous feedback - reward for getting closer to YOLO goals
            reward = 0
            if self.last_distance is not None and self.last_distance != float('inf'):
                distance_improvement = self.last_distance - goal_distance
                reward = distance_improvement * self.continuous_scale
                
                # Extra bonus for being very close to YOLO goals
                if goal_distance < 3.0:
                    proximity_bonus = (3.0 - goal_distance) / 3.0 * 0.02
                    reward += proximity_bonus
            
            self.last_distance = goal_distance
            return 'SmoothDistanceReward', reward
        else:
            # End-of-episode penalty based on distance to YOLO goal
            return 'SmoothDistanceReward', self.reward_factor * goal_distance if state['stop_reasons'] else 0

class CarProximityPenalty(GenericReward):
    def __init__(self, penalty_distance=2.0, max_penalty=-0.05, penalty_scale=2.0, exploration_bonus=0.005):
        super().__init__()
        self.penalty_distance = penalty_distance
        self.max_penalty = max_penalty
        self.penalty_scale = penalty_scale
        self.exploration_bonus = exploration_bonus
        self.goals_found_last_step = 0

    def get_digest(self):
        return f'CarProximityPenalty(distance={self.penalty_distance}, max_penalty={self.max_penalty})'

    def get_reward(self, state):
        # Car proximity penalty
        agent_position = state['car']['position']
        min_distance = float('inf')
        
        for env_module in state['environment']:
            static_cars = env_module.get('static_cars', [])
            for static_car in static_cars:
                if hasattr(static_car, 'position'):
                    car_pos = static_car.position
                elif hasattr(static_car, 'origin'):
                    car_pos = static_car.origin
                else:
                    continue
                
                dx = agent_position[0] - car_pos[0]
                dy = agent_position[1] - car_pos[1]
                distance = (dx*dx + dy*dy) ** 0.5
                min_distance = min(min_distance, distance)
        
        # Calculate proximity penalty
        proximity_penalty = 0.0
        if min_distance <= self.penalty_distance:
            penalty_ratio = max(0, (self.penalty_distance - min_distance) / self.penalty_distance)
            proximity_penalty = self.max_penalty * (penalty_ratio ** self.penalty_scale)
        
        # YOLO goal discovery bonus
        current_yolo_goals = 0
        for env_module in state['environment']:
            if env_module.get('name') == 'YOLOGoals':
                current_yolo_goals = len(env_module.get('parking_goals', []))
                break
        
        # discovery_bonus = max(0, current_yolo_goals - self.goals_found_last_step) * 5.0
        # self.goals_found_last_step = current_yolo_goals
        
        # Exploration reward when no YOLO goals found
        exploration_reward = 0
        if current_yolo_goals == 0:
            car_speed = abs(state['car']['speed'])
            exploration_reward = min(car_speed * self.exploration_bonus, 0.01)
        
        # total_reward = proximity_penalty + discovery_bonus + exploration_reward
        total_reward = proximity_penalty + exploration_reward
        return 'Car Proximity', total_reward


class SmoothDrivingReward(GenericReward):
    def __init__(self, steering_penalty_scale=-0.01, speed_reward_scale=0.005):
        super().__init__()
        self.steering_penalty_scale = steering_penalty_scale
        self.speed_reward_scale = speed_reward_scale

    def get_digest(self):
        return f'SmoothDrivingReward(steering_penalty={self.steering_penalty_scale})'

    def get_reward(self, state):
        car_state = state['car']
        
        # Penalize excessive steering (jerky movements)
        steering_penalty = abs(car_state['wheel_angle']) * self.steering_penalty_scale
        
        # Small reward for appropriate speed (not too slow, not too fast)
        speed = abs(car_state['speed'])
        optimal_speed = 2.0  # Adjust based on your speed scale
        speed_reward = max(0, optimal_speed - abs(speed - optimal_speed)) * self.speed_reward_scale
        
        return 'Smooth Driving', steering_penalty + speed_reward
