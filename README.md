# APEX
A modular Simulation & RL Training Framework

## About APEX
APEX is a framework for training reinforcement learning agents on car navigation tasks, with a focus on modularity. 
It features:
- Physics-based car simulation with realistic steering, acceleration and collision detection
- Modular plugin system for environments, rewards, stop conditions and observations
- TD3 reinforcement learning with parallel training capabilities for RL training
- Vision-based perception support, using ResNets for collision avoidance
- Curriculum training capabilities for progressive difficulty training
- Rich performance monitoring & reward tracking

See the page [here](https://deltaqyto.github.io/CarParkingSim/).

## Setup
Clone this repo, install python 3.10+\
Install required packages with `pip install -r requirements.txt` Packages will take up ~1.5gb.\
Additionally, if you have an nvidia gpu, install [torch with cuda](https://pytorch.org/get-started/locally/) to improve training performance


## Getting Started
                        
The usage scripts are pre-configured for an easy start.\
Simply run any of the five start programs:

#### Manual Control
`start_manual_control.py` - Manually control the agent with keyboard controls for testing.\
Use arrow keys and 'r' to regenerate the environment. When run, the configured environment will appear and begin rendering. 
It will keep going until closed.

#### Single Training  
`start_single_learning.py` - Train a fresh model on one environment.\
You can opt to load an existing model, or from scratch. See the section on model training for details on what to look for.

#### Curriculum Training
`start_curriculum_learning.py` - Train a fresh model on successive environments with increasing difficulty.\
You can opt to start from an existing model, or from scratch. Training a curriculum will take longer than a single environment. See the section on model training for details on what to look for.

#### Vision Training
`start_vision_training.py` - Train a vision model to emulate lidar for use in obstacle avoidance.\
You can set the parameter `'show_random': [seconds]` to visualise a random training image every x seconds, as visual indication of what the model is learning from.
You will see a folder 'vision_models' appear in your model directory, with the currently training model periodically leaving checkpoints.\
The console will provide details on model loss. Once this drops below 0.005, it is typically performant enough to use.

#### Model Testing
`start_test_model.py` - See a pre-trained model navigate the environment, or evaluate trained models with statistics\
Turn off rendering on line 6 to speed up evaluation if you just want the performance figures.

Training and testing will ask for a model 'code' (the model ID). Models appear in the models folder.
Type the code `eg: 'models/td3_STQ4' -> STQ4`.\
Use this code to continue training existing models.
Leave blank when training fresh models.

### Model Training Outputs
When the simulation environment is run, the first thing to be printed (and available in the model folder) is the environment digest.\
The digest is a string that captures the full training environment setup, such as what modules and configs were used.\
This can later be utilised to re-create an environment without needing to share the environment setup code.

When a model is training, you will receive some output on the console, showing time elapsed and model losses.
More useful statistics are displayed within the model's output folder (`models/td3_[code]`).\
Look for an image named `plot.png` showing:
- **Top Left** - Why episodes ended, normalised (collision, timeout, success)
- **Top Right** - Average steps and rewards over time  
- **Bottom Left** - Distance to goal over time
- **Bottom Right** - Individual reward component breakdown

The graphs update every checkpoint (~20k timesteps default). If it appears to not be updating, your image viewer may not be refreshing when the plot is updated.

**Model files use ~1gb per training run with default settings.**\
Consider purging checkpoints after runs

### Post Training
After training, it may be useful to re-plot the training results, performing more iterations per checkpoint. This greatly cuts down on noise.
Simply run the file `start_replotter.py` with the appropriate environment settings used in training, and a new plot will be added to the model folder.

### Performance

Use `performance_test.py` to find optimal environment thread count. This will print out the ideal number of environments to train across.

Default of 14 parallel environments, for good desktop systems.
Reduce if experiencing lag or on a laptop. Increasing above 14 environments is not recommended, as the extra experience threads do not contribute useful experiences.

### Troubleshooting

**"No models found"** - Check model code matches folder name in models. Do not include the `'td3_'` prefix \
**Significant lag during training** - Reduce `num_envs` parameter to free up some cores \
**Low FPS during inference** - Turn off vision if possible (requires disabling vision models as well) \
**Out of memory** - Reduce `buffer_size` or `batch_size` in the parameters \

## Next steps
If you want to start tweaking the environment or training parameters, there are some places to look:
- **Modules**: The modules folder exposes the modules used to create and train models. You can write your own based on the provided.
- **Environments**: `Simulation\environments.py` Shows how to compose modules with the simulation environment to produce a full training environment
- **Curriculum**: `Simulation\training_schedule.py` Shows how to stack environments to make a training curriculum for a potential model.

## Architecture
![Architecture](Images\Architecture.png)
The APEX system implements custom gyms (called simulation environments) and trainers. 
Simulation environments internally handle some aspects of operations, like the car, collision, state management etc.\
Modules are then composed on top of this environment to specialise it.

### Module System
All modules inherit from base classes in `modules/generic_modules.py`:

- **Environment Modules** - Create world geometry, obstacles
- **Reward Functions** - Define training objectives
- **Stop Conditions** - Determine episode termination
- **Observation Modules** - Process sensor data for agent

To make new environments or tasks, it is encouraged to develop separable modules that can be composed, as opposed to a monolithic approach.\
Modules are added at instantiation of the simulation environment, and are expected to remain for its lifetime.

### Unified State Dictionary
Every simulation component receives the same state dictionary containing all simulation data:
```python
state = {
    'car': car_data, 'obstacles': obstacles, 'raycasts': sensor_data,
    'stop_reasons': [], 'reward_types': {}, 'collision_list': [],
    'closest_goal': goal_info, 'environment': env_modules, ...
}
```
**Why**: Any module can access any simulation data without rigid interfaces. Want car position in a reward function? Just use `state['car']['position']`. Need raycast data in a stop condition? Access `state['raycasts']`.

All modules receive state in their main methods (`get_reward()`, `check_stop()`, `get_observation()`). See how `DistanceReward` accesses `state['closest_goal']['distance']` or how `CollisionStop` checks `state['collisions']`.\
Some, like Vision observation functions also mutate the state by overwriting raycasts. This is typically not recommended, but may be necessary.

Modules also expose a function to get their own state. This is packaged and added to the environment state and is used, for example, to let environment modules spawn components relative to other modules
### Digest System
Every module implements `get_digest()` returning a human-readable string of its configuration:
```python
def get_digest(self):
    return f"GoalEndReward(reward={self.reward})"
```
**Why**: Enables easy sharing and reproduction of experimental setups. Share a digest string instead of code files to recreate an environment. Much easier than serialising object hierarchies.

Training saves environment digests to `digest.txt`. Curriculum schedules use digests for documentation. Useful for reproducing experiments from paper descriptions.

### Reward Design

The framework tends to use sparse primary rewards with auxiliary components for exploration.\
In the example environments, the following are used:\
**Primary Objective**: Large sparse reward (+20) for goal completion. Binary success/failure mirrors real-world outcomes.

**Auxiliary Components**: Small continuous rewards help initial exploration:
- `TimePenalty` (-0.01/step) - Encourages efficiency
- `DistanceReward` (-distance/6) - Guides toward goals  
- `CollisionPenalty` (-10) - Avoids crashes
- `ObstacleProximityReward` (-0.02 when near obstacles) - Maintains safe margins

**Natural Curriculum**: As agents improve, the large goal reward dominates auxiliary components. Early training relies on distance gradients while mature agents operate primarily on sparse rewards.


## Creating Environments

### Basic Setup
```python
from Simulation.environments import load_env
from modules.environment_modules import Borders
from modules.reward_functions import GoalEndReward, TimePenalty
from modules.stop_conditions import StepLimit, CollisionStop

environment_modules = [Borders()]
reward_functions = [GoalEndReward(), TimePenalty()]  
stop_conditions = [StepLimit(200), CollisionStop()]

env = load_env(environment_modules=environment_modules, 
               reward_functions=reward_functions,
               stop_conditions=stop_conditions) # Load env is used to turn an environment instance into a factory
```

### Custom Modules
Inherit from generic base classes:
```python
class CustomReward(GenericReward):
    def get_reward(self, state):
        return 'CustomReward', reward_value

class CustomEnvironment(GenericEnvironment):  
    def reset(self, mode, state):
        # Setup environment
        # Mode is a string which tells you whether the module is being reset as an environment, stop condition etc.
        # Helps if you have a conjoined environment-stop condition module, for example
    def render(self, screen, transform):
        # Draw to screen
        # The transform is a 3x3 homogenous matrix which should be applied to anything that lives in world space
        # Naturally, doesn't apply to any hud elements which should be in screen space
    def get_digest(self):
        # return a string that describes the inputs to this function, sufficent to reproduce it
        return f"CustomEnvironment(parameter_1={parameter_1} ...)"
```

## Curriculum Learning

Examples provided in `training_schedule.py`:
```python
class MySchedule(GenericTrainingSchedule):
    def __init__(self):
        self.environments = [easy_env, medium_env, hard_env]  # Environment factories. Do not provide environment instances
        self.parameters = [{'a': 1}, {'b': 1}, {'a': 2}] # The schedule 'stacks' parameters. 
        
        # The easy env receives params: {'a': 1}
        # The medium env receives previous params with the new b field : {'a': 1, 'b': 1}
        # The hard env receives updates a, so it gets: {'a': 2, 'b': 1}.
        # The parameters are training hyperparameters. They are not accessible by the environment
        # Refer to the example or start functions to see what parameters might be required
```

## Vision Training

Train the Raycast Vision model to predict raycasts from images:
1. Train an RL model with raycasts first. See above, or use a pretrained model.
   1. This is done so the vision model learns from a realistic distribution. 
   2. You could instead train on random actions, but doing so tends to collect less valuable training data as the agent is less frequently near obstacles
2. Use `start_vision_training.py` to collect image/raycast pairs. This uses the pre-trained driving agent, and trains the vision model.
   1. The trainer collects image - raycast data pairs from a constantly running environment and learns from it.
   2. The model caches some recent training pairs and trains on a shuffled set of them
   3. Therefore, there is no fixed 'dataset' and no validation set. Current model performance largely indicates actual performance.
3. The vision model is trained to estimate the raycast observation component (observation slice [4:16]) based on an image of the car's surroundings
4. When training concludes, or is otherwise terminated, you can use the trained vision model by adding a `VisionRaycastObservation(vision_model)` to the observation modules
   1. When adding this module, you need to include the classical observation, as the remaining observation is not filled in by the raycast model
   2. Example: `[ClassicalObservation(), VisionRaycastObservation(vision_model)]`, where `vision_model` is the string name of the vision model (eg, "VIS_003c")
