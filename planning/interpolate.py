import carla
from srunner.tools.route_manipulation import interpolate_trajectory

def interpolate(trajectory):
    client = carla.Client('localhost', 2000)
    client.set_timeout(20)
    world = client.get_world()
    gps_route, route = interpolate_trajectory(world, trajectory)
    return gps_route, route
