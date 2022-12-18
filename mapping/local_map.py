import cv2 
import numpy as np
import io
# a = cv2.imread("C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner/srunner/autoagents/myagent/test2.png")
# a=np.array(a)/
# print(mes)
# cv2.imshow('',img)
# cv2.waitKey(10000)
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from visual.distance_detection.yolov5.detect import run
from planning.block_detect import blockdetect

def local_mapping(img=None,
                route_dis=None, 
                route_angle=None,
                traffic_light=None):
    
    o_d=o_theta=o_color=r_d=r_theta=t_d=t_theta=t_color=[]
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.set_thetamin(45.0)  # 设置极坐标图开始角
    ax.set_thetamax(135.0)  # 设置极坐标结束角度
    ax.set_rlabel_position(30.0)  # 标签显示
    ax.set_rgrids(np.arange(0, 20.0, 5.0))
    ax.set_rlim(0.0, 20.0)  # 标签范围
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')  
    
    ax.set_title('yellow-truck\nred-person\ngreen-car\ncircle-traffic light', 
             pad=-5,
             fontsize = 14)
    #plot object
    if img.any():
        img,o_theta,o_d,o_color = plot_object(img)
        ax.scatter(o_theta,o_d,c=o_color,marker=',', s=10**2, cmap='cool', alpha=0.75)

    #plot route
    if route_dis:
        r_theta,r_d=plot_route(route_dis,route_angle)
        ax.scatter(r_theta,r_d,c='black',s=2**2)
    #plot traffic light
    if traffic_light:
        t_theta,t_d,t_color=plot_traffic_light(traffic_light)
        ax.scatter(t_theta, t_d, c=t_color)
    situation = [o_d,o_theta,o_color,r_d,r_theta,t_d,t_theta,t_color]
    plot_img_np = get_img_from_fig(fig)

    return img,plot_img_np,situation
        
    
    
def plot_object(img):
    img, mes = run(img=img)#object_detection
    _,size,_ = img.shape
    r = []
    theta = []
    color=[]
    for i in mes:
        new = ''.join([i for i in i[2] if not i.isdigit()])
        new=new.replace('.','')
        new=new.replace(' ','')
    
        if new!='car' and new!='truck'and new!='person'and new!='motorcycle':
            break
        angle, dis= i[0],i[1]
        angle = 0.9*np.pi-angle.cpu().numpy()*(np.pi*0.8)/(size)

        if angle<(np.pi*0.78) and angle>(np.pi*0.22):
            dis = dis.cpu().numpy()/4#5
            theta.append(angle)
            r.append(dis)

            # label
            if new=='car':
                color.append('g')
            elif new=='motorcycle':
                color.append('g')
            elif new=='truck':
                color.append('c')
            elif new=='person':
                color.append('blue')
    return img,theta, r,color


    
    
#plot route
def plot_route(route_dis,route_angle):
    if route_dis and route_angle:
        theta=[i*np.pi/180 for i in route_angle]
        r=[i*0.65 for i in route_dis]
        return theta,r


#plot traffic light
def plot_traffic_light(traffic_light):
    if traffic_light:
        theta = []
        r=[]
        color=[]
        for light in traffic_light:
            angle, dis = light[0],light[1]
            angle = 0.625*np.pi-angle*(np.pi*0.25)/1200
            dis = 70*dis
            # print(f"d={dis}"+light[2])
            theta.append(angle)
            r.append(dis)
            # print(angle)
            # print(dis)
        
        # label
            if light[2]=='red':
                color.append('blue')
            elif light[2]=='yellow':
                color.append('orange')
            elif light[2]=='green':
                color.append('green')
    return theta, r, color



def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
