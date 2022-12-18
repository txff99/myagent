from .local_map import get_img_from_fig
import cv2 
import math
import numpy as np
import matplotlib.pyplot as plt
from .location2np import carla_location_to_numpy_vector
from .location2np import carla_rotation_to_numpy_vector


def global_mapping(route,current,local_route):
    fig = plt.figure()
    x=[]
    y=[]

    #global route 
    for i in route:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y)
    x=[]
    y=[]
    #local route
    for i in local_route:
        x.append(i[0])
        y.append(i[1])
    
    plt.scatter(x, y,color = 'b')
    #current location
    location = carla_location_to_numpy_vector(current.location)
    rotation = carla_rotation_to_numpy_vector(current.rotation)
    plt.scatter(location[0],location[1],color = 'r')
    dis,angle = global2local(location, rotation, local_route)
    

    plot_img_np = get_img_from_fig(fig)
    # cv2.imshow('',plot_img_np)
    # cv2.waitKey(50)
    return dis, angle, plot_img_np

def global2local(location, rotation, local_route):
    dis = []
    angle=[]
    for i in local_route:
        dis.append(math.sqrt((i[0]-location[0])**2+(i[1]-location[1])**2)*0.7)
        angle.append(180-math.atan2((i[0]-location[0]),(i[1]-location[1]))*180/np.pi+rotation[1])
    # print(angle[0])
    # print(dis[0])
    return dis,angle