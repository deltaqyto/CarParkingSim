from modules.generic_modules import GenericEnvironment
from Objects.obstacles import RectObstacle
from Objects.car import Car


# Change the configuration of the environment by changing the configuration parameter
class Borders(GenericEnvironment):
    def __init__(self, wall_width=2, configuration=0):
        super().__init__()
        self.configuration = configuration
        self.wall_width = wall_width
        self.world_width = None
        self.world_height = None
        self.collision_rects = []
        self.static_cars = []
        self.obstacles = []

    def reset(self, mode, state=None):
        # Environment setup
        world_size = state['world_size']
        self.world_width = world_size[0]
        self.world_height = world_size[1]
        self.collision_rects = [RectObstacle([0, self.world_height/2 - self.wall_width/2], [self.world_width, self.wall_width]),
                            RectObstacle([0, -self.world_height/2 + self.wall_width/2], [self.world_width, self.wall_width]),
                            RectObstacle([self.world_width/2 - self.wall_width/2, 0], [self.wall_width, self.world_height]),
                            RectObstacle([-self.world_width/2 + self.wall_width/2, 0], [self.wall_width, self.world_height])]

        print("Adding static cars.. ")
        
        if self.configuration == 0:
        
            self.static_cars = [
            Car(origin=[-22, 6], start_direction=0, color=(240, 230, 0)),
            Car(origin=[-22, 0], start_direction=0, color=(0, 255, 0)),
            Car(origin=[-22, -6], start_direction=0, color=(0, 233, 255)),
            Car(origin=[-22, -12], start_direction=0, color=(0, 255, 0)),
            Car(origin=[-22, 12], start_direction=0, color=(0, 233, 255)),
            
            Car(origin=[22, 6], start_direction=0, color=(240, 230, 0)),
            Car(origin=[22, 0], start_direction=0, color=(0, 255, 0)),
            Car(origin=[22, -6], start_direction=0, color=(0, 233, 255)),
            Car(origin=[22, -12], start_direction=0, color=(0, 255, 0)),
            Car(origin=[22, 12], start_direction=0, color=(0, 233, 255))
            ]
            
            self.obstacles = [
                RectObstacle([-22, 15], [4.7, 2]),
                RectObstacle([-22, 3], [4.7, 2]),
                RectObstacle([-22, -3], [4.7, 2]),
                RectObstacle([-22, -9], [4.7, 2]),
                RectObstacle([-22, 9], [4.7, 2]),
                RectObstacle([-22, -15], [4.7, 2]),
                
                RectObstacle([22, 15], [4.7, 2]),
                RectObstacle([22, 3], [4.7, 2]),
                RectObstacle([22, -3], [4.7, 2]),
                RectObstacle([22, -9], [4.7, 2]),
                RectObstacle([22, 9], [4.7, 2]),
                RectObstacle([22, -15], [4.7, 2]),
                # RectObstacle([-22, 14], [4.7, 2])
            ]
            
        elif self.configuration == 1:
            
            self.static_cars = [
            Car(origin=[-22, 10], start_direction=90, color=(240, 230, 0)),
            Car(origin=[-16, 10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[-10, 10], start_direction=90, color=(0, 233, 255)),
            Car(origin=[-4, 10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[2, 10], start_direction=90, color=(0, 233, 255)),
            Car(origin=[8, 10], start_direction=90, color=(240, 230, 0)),
            Car(origin=[14, 10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[20, 10], start_direction=90, color=(0, 233, 255)),
            
            Car(origin=[-22, -10], start_direction=90, color=(240, 230, 0)),
            Car(origin=[-16, -10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[-10, -10], start_direction=90, color=(0, 233, 255)),
            Car(origin=[-4, -10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[2, -10], start_direction=90, color=(0, 233, 255)),
            Car(origin=[8, -10], start_direction=90, color=(240, 230, 0)),
            Car(origin=[14, -10], start_direction=90, color=(0, 255, 0)),
            Car(origin=[20, -10], start_direction=90, color=(0, 233, 255))
            
            
            ]
            
            self.obstacles = [
                RectObstacle([-19, 10], [4.7, 2],angle=90),
                RectObstacle([-13, 10], [4.7, 2],angle=90),
                RectObstacle([-7,  10], [4.7, 2],angle=90),
                RectObstacle([-1, 10], [4.7, 2],angle=90),
                RectObstacle([5, 10], [4.7, 2],angle=90),       
                RectObstacle([11, 10], [4.7, 2],angle=90),
                RectObstacle([17, 10], [4.7, 2],angle=90),
                RectObstacle([23, 10], [4.7, 2],angle=90),
                
                

                RectObstacle([-19, -10], [4.7, 2],angle=90),
                RectObstacle([-13, -10], [4.7, 2],angle=90),
                RectObstacle([-7, -10], [4.7, 2],angle=90),
                RectObstacle([-1, -10],  [4.7, 2],angle=90),
                RectObstacle([5, -10],   [4.7, 2],angle=90),
                RectObstacle([11, -10],   [4.7, 2],angle=90),
                RectObstacle([17, -10],  [4.7, 2],angle=90),
                RectObstacle([23, -10],  [4.7, 2],angle=90)

            ]
        
        
    def render(self, screen, transform_matrix):
        for rect in self.collision_rects:
            rect.draw(screen, transform_matrix)
            
        for car in self.static_cars:
            car.draw(screen, transform_matrix)
            
        for obstacle in self.obstacles:
            obstacle.draw(screen, transform_matrix)

    def get_digest(self):
        return f"Borders(world_width={self.world_width}, world_height={self.world_height}, "\
               f"wall_width={self.wall_width})"

    def get_unified_state(self):
        return {'obstacles': self.collision_rects,'static_cars': self.static_cars, 'static_obstacles': self.obstacles}