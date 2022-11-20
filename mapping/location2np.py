import numpy
def carla_location_to_numpy_vector(carla_location):
    """
    Convert a carla location to a ROS vector3
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS)
    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    return numpy.array([
        carla_location.x,
        -carla_location.y,
        carla_location.z
    ])

def carla_rotation_to_numpy_vector(rotation):
    return numpy.array([
        rotation.pitch,
        rotation.yaw,
        rotation.roll
    ])