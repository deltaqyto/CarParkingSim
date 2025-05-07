import numpy as np


class Ray:
    def __init__(self, origin, direction):
        # Ensure origin and direction are the right dimensionality
        self.origin = np.array(origin, dtype=float)
        # Store both normalized and original direction (for performance)
        self.raw_direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        # Avoid division by zero
        if norm > 1e-10:
            self.direction = self.raw_direction / norm
        else:
            self.direction = np.array([1.0, 0.0])  # Default direction if input is zero


def ray_cast(ray, obstacles, max_distance=10):
    """Cast ray against obstacles, return distance, hit point, and hit object"""
    closest_hit = max_distance
    hit_point = None
    hit_object = None

    # Quick check for empty obstacles list
    if not obstacles:
        endpoint = ray.origin + ray.direction * max_distance
        return max_distance, endpoint, None

    # Pre-compute ray endpoint for max distance
    ray_end = ray.origin + ray.direction * max_distance

    # Process obstacles in batches for better cache locality
    for obj in obstacles:
        # Fast AABB rejection test
        aabb = obj.get_aabb()  # This is now cached

        # Quick AABB check
        if not _ray_aabb_intersection(ray, aabb, max_distance):
            continue

        # If AABB check passes, do detailed rectangle intersection
        distance, point = _intersect_rectangle(ray, obj)

        if distance is not None and distance < closest_hit:
            closest_hit = distance
            hit_point = point
            hit_object = obj

    # If we hit something, return the details
    if hit_point is not None:
        return closest_hit, hit_point, hit_object
    else:
        # Calculate endpoint at max distance
        endpoint = ray.origin + ray.direction * max_distance
        return max_distance, endpoint, None


def _ray_aabb_intersection(ray, aabb, max_distance):
    """
    Fast check if ray intersects with axis-aligned bounding box
    aabb format: (min_x, min_y, max_x, max_y)
    """
    min_x, min_y, max_x, max_y = aabb

    # Ray origin
    ox, oy = ray.origin
    # Ray direction
    dx, dy = ray.direction

    # Very small value to avoid division by zero
    epsilon = 1e-10

    # Calculate t values for each AABB plane
    if abs(dx) < epsilon:
        # Ray is parallel to y-axis
        if ox < min_x or ox > max_x:
            return False
        # Force valid t values
        tx_min, tx_max = -float('inf'), float('inf')
    else:
        # Calculate intersections with x-planes
        tx_min = (min_x - ox) / dx
        tx_max = (max_x - ox) / dx
        # Ensure tx_min <= tx_max
        if tx_min > tx_max:
            tx_min, tx_max = tx_max, tx_min

    if abs(dy) < epsilon:
        # Ray is parallel to x-axis
        if oy < min_y or oy > max_y:
            return False
        # Force valid t values
        ty_min, ty_max = -float('inf'), float('inf')
    else:
        # Calculate intersections with y-planes
        ty_min = (min_y - oy) / dy
        ty_max = (max_y - oy) / dy
        # Ensure ty_min <= ty_max
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

    # Find the max of the min values and min of the max values
    t_enter = max(tx_min, ty_min)
    t_exit = min(tx_max, ty_max)

    # Check if there's a valid intersection within max_distance
    return t_exit >= t_enter and t_enter < max_distance and t_exit > 0


def _intersect_rectangle(ray, rect_obj):
    """Check ray intersection with rectangle object - optimized version"""
    # Get rectangle corners (now cached)
    corners = rect_obj.get_corners()

    # Pre-allocate min distance and hit point
    min_dist = None
    hit_point = None

    # Use numpy arrays throughout for better performance
    segments = []
    for i in range(len(corners)):
        segments.append((corners[i], corners[(i + 1) % len(corners)]))

    # Process segments
    for p1, p2 in segments:
        # Check intersection with this segment
        dist, point = _intersect_segment(ray, p1, p2)

        if dist is not None and (min_dist is None or dist < min_dist):
            min_dist = dist
            hit_point = point

    return min_dist, hit_point


def _intersect_segment(ray, p1, p2):
    """Check ray intersection with line segment - optimized version"""
    # Create segment vector
    segment_dir = p2 - p1

    # Pre-compute cross product for intersection test
    perp_dot = ray.direction[0] * segment_dir[1] - ray.direction[1] * segment_dir[0]

    # Early out if parallel (within tolerance)
    if abs(perp_dot) < 1e-10:
        return None, None

    # Vector from ray origin to segment start
    origin_to_p1 = p1 - ray.origin

    # Calculate u parameter for segment intersection
    u = (origin_to_p1[0] * ray.direction[1] - origin_to_p1[1] * ray.direction[0]) / perp_dot

    # Check if intersection is within segment bounds
    if u < 0 or u > 1:
        return None, None

    # Calculate t parameter for ray intersection
    t = (origin_to_p1[0] * segment_dir[1] - origin_to_p1[1] * segment_dir[0]) / perp_dot

    # Check if intersection is in front of the ray
    if t < 0:
        return None, None

    # Calculate intersection point
    hit_point = ray.origin + t * ray.direction
    return t, hit_point