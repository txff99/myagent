#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function
from concurrent.futures import process
import cv2
import carla
from agents.navigation.basic_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from visual.traffic_light.TrafficLight_Detector.src.main import light
from mapping.local_map import local_mapping,get_img_from_fig
from mapping.global_map import global_mapping
from mapping.np2location import numpy2loc
from mapping.location2np import carla_location_to_numpy_vector
from planning.lane_change import change_lane
from planning.local_planner import LocalPlanner
from planning.interpolate import interpolate
from agents.tools.misc import get_speed
import numpy as np
import matplotlib.pyplot as plt

class DummyAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.route=[]
        self._route_assigned = False
        # self._agent = None
        self.trajectory = []
        self.plan = []
        self.time=0
        self.change_lane=False
        self.vehicle = None
        self.planner = None
        self.situation = []
        self.time=0
        self.dis=self.angle=self.global_map=[]
    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


        """

        sensors = [
            # {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left','shutter_speed': 5.0},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': -5.0, 'yaw': 0.0,
             'width': 800, 'height': 600, 'fov': 100, 'id': 'Middle'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 10.0, 'yaw': 0.0,
             'width': 400, 'height': 300, 'fov': 15, 'id': 'row'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': -0.0, 'pitch': 10.0, 'yaw': 15.0,
             'width': 400, 'height': 300, 'fov': 15, 'id': 'yawright'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 10.0, 'yaw': -15.0,
             'width': 400, 'height': 300, 'fov': 15, 'id': 'yawleft'}
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        print("=====================>")
        self.time+=1     
        a = input_data["Middle"][1][:,:,:3]  
        r=input_data["yawright"][1][:,:,:3]
        l=input_data["yawleft"][1][:,:,:3]
        m=input_data["row"][1][:,:,:3]

        com = np.concatenate((l,m,r),axis=1)
    
        # cv2.imshow('middle',a)
        # cv2.waitKey(50)

        li = light()

        print("<=====================")

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self.planner:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            # if hero_actor:
            #     self._agent = BasicAgent(hero_actor)
            if hero_actor:
                self.vehicle = hero_actor
                self.planner = LocalPlanner(hero_actor)

            return control
        if not self._route_assigned:


        #set the global plan
            if self._global_plan:
                self.trajectory=[]
                self.route=[]
                self.plan=[]
                
   
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    self.plan.append((wp, road_option))
                    self.route.append(carla_location_to_numpy_vector(transform.location))
                for transform in self._config_trajectory:
                    wp = np.array([transform.x,-transform.y,transform.z])
                    self.trajectory.append(wp)
                
                self.planner.set_global_plan(self.plan)  
                self._route_assigned = True
                
        
        else:
            pack = self.planner.run_step(self.situation)
            
            if len(pack)>1:
                control,location,local_route,self.change_lane=pack
            
            #------------delete passed trajectory-----------
                
                k=1.5
                j = location.location
                i = self.trajectory[0]
                #if current location pass the trajectory, delete the point
                if j.x-k<=i[0]<=j.x+k and j.y-k<=-i[1]<=j.y+k and j.z-k<=i[2]<=j.z+k:
                    self.trajectory=self.trajectory[1:]
                #if location path pass the trajectory, delete the point
                j=local_route[-1]
                if j[0]-k<=i[0]<=j[0]+k and j[1]-k<=-i[1]<=j[1]+k and j[2]-k<=i[2]<=j[2]+k:
                    self.trajectory=self.trajectory[1:]

                self.dis,self.angle,self.global_map = global_mapping(self.route,location,local_route)
                
            #----------------change lane--------------------
                
                waypoint=[]
                if self.change_lane:
                    #self.trajectory=self.trajectory[1:]
                    for i in local_route[3:]:
                        waypoint.append(change_lane(CarlaDataProvider.get_map().get_waypoint(numpy2loc(i))))
                    if waypoint:
                    # change = change_lane(waypoint)
                    # print(change)
                    # if change_lane:
                        new_trajectory=[]
                        for i in self.trajectory:
                            new_trajectory.append(numpy2loc(i))

                        new_trajectory=waypoint+new_trajectory
                        # print(new_trajectory)
                        # print(location.location)
                        # for i in local_route:
                        #     print(i)
                        # print(11111111111111111)
                        # for i in new_trajectory:
                        #     print(i)
                        # print(1111111111111111111111111111111111111111111111)
                        # for i in self._config_trajectory:
                        #     print(i)
                        gps, rt = interpolate(new_trajectory)
                        self.set_global_plan(gps,rt,new_trajectory)
                        self._route_assigned=False
                        self.change_lane=False
            else:
                control = pack[0]


#---------------plot-------------------
            if self.time==3: # 4 frames
                self.time=0
                # lightdetect = li.light_detect(com)

                img,map,self.situation = local_mapping(img=a,
                                                    route_dis=self.dis,
                                                    route_angle=self.angle)
                                                    #traffic_light=li.light_position)

                fig = plt.figure(figsize=(8,8))
                fig.suptitle(f'speed:{get_speed(self.vehicle):.2f}km/h \n throttle:{control.throttle:.2f} \n brake:{control.brake:.2f} \n steering:{control.steer:.2f}')
                ax=plt.subplot(2,2,1)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.set_title("object detection")
                plt.imshow(img)
                ax=plt.subplot(2,2,2)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.set_title("local map")
                plt.imshow(map)
                # ax=plt.subplot(2,2,3)
                # ax.get_yaxis().set_visible(False)
                # ax.get_xaxis().set_visible(False)
                # ax.set_title("traffic light detection")
                # plt.imshow(lightdetect)
                ax=plt.subplot(2,2,4)
                ax.set_title("global map")
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                plt.imshow(self.global_map)
                # plt.show()
                im = get_img_from_fig(fig)
                cv2.imshow('11',im)
                cv2.waitKey(20)
                # save_dir = f'C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner/srunner/autoagents/myagent/video/img_{timestamp}.png'
                # status = cv2.imwrite(save_dir,im)
                # print(status)




        

        return control

# set CARLA_ROOT=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor &&set LEADERBOARD_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboardset &&set SCENARIO_RUNNER_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner &&set PYTHONPATH=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner;C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg &&cd C:/Users/22780\Documents\CARLA_0.9.13\leaderboard\scenario_runner &&conda activate py37 &&python scenario_runner.py --route srunner/data/routes_devtest.xml srunner/data/all_towns_traffic_scenarios.json 8 --agent srunner/autoagents/myagent/npc_agent.py --debug 

# set CARLA_ROOT=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor &&set LEADERBOARD_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboardset &&set SCENARIO_RUNNER_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner &&set PYTHONPATH=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner;C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg &&cd C:\Users\22780\Documents\CARLA_0.9.13\leaderboard\scenario_runner &&conda activate py37 &&python scenario_runner.py --route srunner/data/routes_debug.xml srunner/data/all_towns_traffic_scenarios1_3_4.json 0 --agent srunner/autoagents/myagent/npc_agent.py