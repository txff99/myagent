
import numpy as np
class blockdetect(object):
    def __init__(self,situations=[]):
        if len(situations)>1:
            self.state=True
            self.o_d = situations[0]
            self.o_ang = situations[1]
            self.o_type = situations[2]
            self.r_d = situations[3]
            self.r_ang = situations[4]
            self.t_d = situations[5]
            self.t_ang = situations[6]
            self.t_type = situations[7]
        else:
            self.state=False
        

    def object_block(self):
        if self.state:
            emergency = []
            dis = []
            for i in range(len(self.o_d)):
                for j in range(len(self.r_d)):
                    dis.append(np.sqrt((self.o_d[i]*np.cos(self.o_ang[i])-self.r_d[j]*np.cos(self.r_ang[j]))**2\
                        +(self.o_d[i]*np.sin(self.o_ang[i])-self.r_d[j]*np.sin(self.r_ang[j]))**2)) #distance between the object and the path
                block = min(dis)
                # print(f"block{block}")
                if 0<=self.o_d[i] <=10 and self.o_type[i]=="blue":
                    emergency.append(1)
                elif 6<=self.o_d[i] <=10 and block<1.3 and self.o_type[i]!="blue":
                    emergency.append(3)
                elif 0<=self.o_d[i] <=6 and block<1.6 and self.o_type[i]!="blue":
                    emergency.append(2)
                elif 10<=self.o_d[i]<=20 and np.pi/2-0.1<self.o_ang[i]<np.pi/2+0.1 and self.o_type[i]!="blue":
                    emergency.append(4)
                dis=[]
            
            else:
                emergency.append(5)
            if 1 in emergency:
                return "person block"
            elif 2 in emergency:
                return "short distance"
            elif 3 in emergency:
                return "middle distance"
            elif 4 in emergency:
                return "long distance"
            else:
                return "no engage"
        else:
            return "no engage"
        
    def light_detect(self):
        if self.state:
            for i in range(len(self.t_d)):
                # print(self.t_d[i])
                # print(self.t_type[i])
                if 0<=self.t_d[i] <=15 and self.t_type[i]=="blue":
                    return "red light"
                else:
                    return "no engage"
        else:
            return "no engage"

                
