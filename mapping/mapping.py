from visual.distance_detection.yolov5.detect import run
import cv2 
import numpy as np
import io
# a = cv2.imread("C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner/srunner/autoagents/myagent/test2.png")
# a=np.array(a)/
# print(mes)
# cv2.imshow('',img)
# cv2.waitKey(10000)
import matplotlib.pyplot as plt


def mapping(a,route_dis, route_angle):
    

    img, mes = run(img=a)
    # print(a.shape)
    _,size,_ = a.shape
    r = []
    theta = []
    color=[]
    fig = plt.figure()
    for i in mes:
        
        new = ''.join([i for i in i[2] if not i.isdigit()])
        new=new.replace('.','')
        new=new.replace(' ','')
        # print(new)
        if new!='car' and new!='truck'and new!='person'and new!='motorcycle':
            break
        angle, dis= i[0],i[1]
        angle = 0.9*np.pi-angle.cpu().numpy()*(np.pi*0.8)/(size)
        
        # print(angle)
        
        # print(type(angle))
        # if 0.33*np.pi>angle:
        #     _,mesr=run(img=right)
        #     print(1)
        #     for r in mesr:
        #         angle, dis= i[0],i[1]
        #         angle = 0.33*np.pi-angle.cpu().numpy()*(np.pi*0.5)/(size)
        #         dis = dis.cpu().numpy()/5
        #         print(angle)
        #         print(dis)
        #         theta.append(angle)
        #         r.append(dis)
        # # if angle>0.67*np.pi: 
        # #     run(img=left)
        # elif angle>0.67*np.pi:
        #     print(2)
        #     _,mesl=run(img=left)
        #     for r in mesl:
        #         angle, dis= i[0],i[1]
        #         angle = 0.33*np.pi-angle.cpu().numpy()*(np.pi*0.5)/(size)
        #         dis = dis.cpu().numpy()/5
        #         print(angle)
        #         print(dis)
        #         theta.append(angle)
        #         r.append(dis)
        # else:
        #     print(3)

        dis = dis.cpu().numpy()/2
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
        # else:
        #     color.append('k')

    # print(j)
    # plt.polar(i)

    area = 7**2 #面积
    # colors = theta #颜色
    ax = plt.subplot(111, projection='polar')
#projection为画图样式，除'polar'外还有'aitoff', 'hammer', 'lambert'等
    ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
    ax.set_thetamax(180.0)  # 设置极坐标结束角度为180°
    # ax.set_rlabel_position(0.0)  # 标签显示在0°
    ax.set_rlim(0.0, 20.0)  # 标签范围为[0, 5000)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
    ax.set_title('yellow-truck\nblue-person\ngreen-car', 
             pad=-5,
             fontsize = 14)
    
    print()
    ax.scatter(theta, r,c=color, s=area, cmap='cool', alpha=0.75)
    ax.scatter([i*np.pi/180 for i in route_angle],route_dis,c='black',s=2**2)
    # plt.axis('off')
    plot_img_np = get_img_from_fig(fig)
    return img,plot_img_np
#ax.scatter为绘制散点图函数
# plt.show()
# creating an array
# containing the radian values
# rads = np.arange(0, (2 * np.pi), 0.01)
  
# plotting the ellipse
# ax = fig.add_subplot(111)
# for rad in rads:
    # r = (a*b)/math.sqrt((a*np.sin(rad))**2 + (b*np.cos(rad))**2)
  
# display the polar plot
# plt.show()

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
# cv2.imshow('',plot_img_np)
# cv2.waitKey(5000)

  
  
# setting the axes
# projection as polar
# plt.axes(projection = 'polar')

# setting the values of
# semi-major and
# semi-minor axes

# display the Polar plot