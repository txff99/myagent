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


def local_mapping(img,
                route_dis=None, 
                route_angle=None,
                traffic_light=None):
    

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    #projection为画图样式，除'polar'外还有'aitoff', 'hammer', 'lambert'等
    ax.set_thetamin(45.0)  # 设置极坐标图开始角度为0°
    ax.set_thetamax(135.0)  # 设置极坐标结束角度为180°
    ax.set_rlabel_position(30.0)  # 标签显示在0°
    ax.set_rgrids(np.arange(0, 20.0, 5.0))
    ax.set_rlim(0.0, 20.0)  # 标签范围为[0, 5000)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
    
    ax.set_title('yellow-truck\nblue-person\ngreen-car', 
             pad=-5,
             fontsize = 14)
    
    
    img, mes = run(img=img)
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
            dis = dis.cpu().numpy()/5
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
                color.append('r')


    area = 7**2 #面积
    #plot vehicle
    ax.scatter(theta, r,c=color, s=area, cmap='cool', alpha=0.75)
    
    
    #plot route
    if route_dis and route_angle:
        ax.scatter([i*np.pi/180 for i in route_angle],route_dis,c='black',s=2**2)
    
    
    #plot traffic light
    if traffic_light:
        theta = []
        r=[]
        color=[]
        for light in traffic_light:
            angle, dis = light[0],light[1]
            
            
            
            angle = 0.625*np.pi-angle*(np.pi*0.25)/1200
            dis = 100*dis
            print(f"d={dis}"+light[2])
            theta.append(angle)
            r.append(dis)
        
        # label
            print(light[2])
            if light[2]=='red':
                color.append('r')
            elif light[2]=='y':
                color.append('Yellow')
            elif light[2]=='green':
                color.append('g')
            # elif new=='person':
            #     color.append('r')
        # print(theta)
        # print(color)
        ax.scatter(theta, r, c=color,marker=',')

    # plt.axis('off')
    plot_img_np = get_img_from_fig(fig)
    return img,plot_img_np


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
