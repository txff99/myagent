import carla

def numpy2loc(x):
    return carla.Location(x=x[0],y=-x[1],z=x[2])