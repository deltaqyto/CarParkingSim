from modules.generic_modules import GenericEnvironment
from Objects.obstacles import RectObstacle


class Borders(GenericEnvironment):
    def __init__(self, wall_width=2):
        super().__init__()
        self.wall_width = wall_width
        self.world_width = None
        self.world_height = None
        self.collision_rects = []

    def reset(self, collision_system, state, world_width=60, world_height=40):
        self.world_width = world_width
        self.world_height = world_height
        self.collision_rects = [RectObstacle([0, world_height/2 - self.wall_width/2], [world_width, self.wall_width]),
                            RectObstacle([0, -world_height/2 + self.wall_width/2], [world_width, self.wall_width]),
                            RectObstacle([world_width/2 - self.wall_width/2, 0], [self.wall_width, world_height]),
                            RectObstacle([-world_width/2 + self.wall_width/2, 0], [self.wall_width, world_height])]

        for rect in self.collision_rects:
            collision_system.add_object(rect)

    def render(self, screen, transform_matrix):
        for rect in self.collision_rects:
            rect.draw(screen, transform_matrix)

    def get_digest(self):
        return f"Borders(world_width={self.world_width}, world_height={self.world_height}, "\
               f"wall_width={self.wall_width})"

    def get_unified_state(self):
        return {'obstacles': self.collision_rects}

