from .mapping import get_img_from_fig
import cv2 
import numpy as np
import matplotlib.pyplot as plt


def global_mapping(route,location,local_route):
    x=[]
    y=[]
    fig = plt.figure()
    for i in route:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y)
    x=[]
    y=[]
    for i in local_route:
        x.append(i[0])
        y.append(i[1])
    # print(len(x))
    plt.scatter(x, y,color = 'b')
    print(location[0])
    plt.scatter(location[0],location[1],color = 'r')
    plot_img_np = get_img_from_fig(fig)
    cv2.imshow('',plot_img_np)
    cv2.waitKey(50)
