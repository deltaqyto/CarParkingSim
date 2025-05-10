# Setup
Install required packages with `pip install -r requirements.txt`
Additionally, if you have an nvidia gpu, install torch & cuda to improve training performance

Packages will take up ~1.5gb. 
No other setup should be required

# Getting started
You can run any of the four start programs to see how they function.
* Manual -> Keyboard controls, prints out your reward each episode
* Single learning -> Trains a full model based on the environment
* Curriculum learning -> Trains a full model based on successive environments defined in the curriculum
* Test model -> Playback the model, and get some statistics

Training and testing will ask you for a model 'code'. This is the id of the model. Models will appear in the models folder.
Simply type out the code eg 'models/td3_STQ4' -> 'STQ4'.
You can use this model code to continue training on existing models, if need be.

When the model is training, you can look inside the model folder, and will get a graph that shows:
* Top Left -> Why the model stopped (hit a wall, ran out of time etc). This is averaged over some number of runs
* Top right -> Average model steps & average reward
* Bottom left -> Distance to the nearest goal
* Bottom right -> A breakdown of individual rewards. Useful to see if your model is reward-gaming, or if a reward function is broken 

This graph will update each time the model makes a checkpoint (~20k timesteps)
### Model files will take up a lot of space. ~1gb / training run with default settings

## Environments
Setup an environment, starting in environments.py.
Here, you can define an environment and add modules for the world, reward and episode stop conditions.

Once you have an environment (or are using the example provided), you can import it into the manual, single learning or test scripts.

## Curriculum learning
This codebase supports training curriculums. Here, you can define successive training environments and settings to progressively approach a solution.
Think of this as automatically running the planned environments back to back as each concludes. For example, you could reduce the goal size progressively each time.

To get started, set up your curriculum in training_schedule.py. Refer to the example within.
Once ready, you can start curriculum learning with your curriculum, and leave it to train.
