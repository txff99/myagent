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
from visual.lane_detection.CarLaneDetection.src.lane_detection import lane_detect
# from lane_detection.Advanced_Lane_Lines.pipeline import Pipeline, Line,img_process
# from lane_detection.Driver_Guidance_System.lanes import img_process
from visual.traffic_light.TrafficLight_Detector.src.main import light_detect
# from distance_detection.vehicle_distance.yolo import YOLO
from visual.distance_detection.yolov5.detect import run
from visual.lane_detection.Lane_Detection.Lane_Detection_window import parse_image
from visual.lane_detection.advanced_lane_detection.line_fit_video import annotate_image
from mapping.mapping import mapping
from mapping.global_map import global_mapping
import numpy as np
from mapping.location2np import carla_location_to_numpy_vector
class NpcAgent(AutonomousAgent):

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
        self._agent = None

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
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': -5.0, 'yaw': 0.0,
             'width': 1500, 'height': 500, 'fov': 125, 'id': 'Middle'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0,
             'width': 1000, 'height': 600, 'shutter_speed': 100,'fov': 20, 'id': 'Right'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': -0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 500, 'height': 300, 'shutter_speed': 50,'fov': 100, 'id': 'rowright'},
            {'type': 'sensor.camera.rgb', 'x': 1.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
             'width': 500, 'height': 300, 'shutter_speed': 50,'fov': 100, 'id': 'rowleft'}
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        print("=====================>")
        # cv2.imshow('left',annotate_image(input_data["Left"][1][:,:,:3]))
        # cv2.waitKey(50)
        # a=run(img = input_data["Right"][1][:,:,:3])
        # a=light_detect(input_data["Right"][1][:,:,:3])
        
        # a = input_data["Middle"][1][:,:,:3]  
        # # r=input_data["rowright"][1][:,:,:3]
        # # l=input_data["rowleft"][1][:,:,:3]
        # # stitcher = Stitcher()
        # # paro = stitcher.stitch(a, r, l)
        # img,map = mapping(a)
        # # com = np.concatenate((img),axis=1)
        # cv2.imshow('middle',img)
        # cv2.waitKey(50)
        # cv2.imshow('map',map)
        # cv2.waitKey(50)
        # cv2.imshow('left',com)
        # cv2.waitKey(50)
        
        # save_dir = f'C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner/srunner/autoagents/myagent/video/img_{timestamp}.png'
        # status = cv2.imwrite(save_dir,a)#input_data["Left"][1][:,:,:3])
        # print(status)
        # for key, val in input_data['Left'].items():
        #     if hasattr(val[1], 'shape'):
                # yolo = YOLO()
                # parse_image(val[1][:,:,:3])
                # cv2.imshow('1',annotate_image(val[1][:,:,:3]))
                # cv2.waitKey(50)
                # cv2.imshow('1',run(img=val[1][:,:,:3]))
                # cv2.waitKey(50)
                # out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1000,800))
                # for i in range(len(img_array)):
                #     out.write(img_array[i])
                # out.release()
                # print(type(yolo.detect_image(val[1][:,:,:3])))
                # cv2.imshow('dummy',np.array(yolo.detect_image(val[1][:,:,:3])))               
                # cv2.waitKey(200)
                # shape = val[1].shape
                # print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
            # else:
                # print("[{} -- {:06d}] ".format(key, val[0]))
        print("<=====================")

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor)

            return control
        if not self._route_assigned:
            if self._global_plan:
                plan = []
                
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                    # print(carla_location_to_numpy_vector(transform.location))
                    self.route.append(carla_location_to_numpy_vector(transform.location))
                # print(len(route))
                # print(plan[0][0])
                # print(plan[1][0])
                # print(plan[2][0])
                # print(plan[3][0])
                # print(len(plan))
                # exit()
                
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
        
        else:
            pack = self._agent.run_step()
            # print(pack)
            if len(pack)>1:
                control,location,local_route=pack
                global_mapping(self.route,location,local_route)
            else:
                control = pack[0]


        return control

# set CARLA_ROOT=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor &&set LEADERBOARD_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboardset &&set SCENARIO_RUNNER_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner &&set PYTHONPATH=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner;C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg &&cd C:/Users\22780\Documents\CARLA_0.9.13\leaderboard\scenario_runner &&conda activate py37 &&python scenario_runner.py --route srunner/data/routes_devtest.xml srunner/data/all_towns_traffic_scenarios.json 8 --agent srunner/autoagents/npc_agent.py --debug 

# set CARLA_ROOT=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor &&set LEADERBOARD_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboardset &&set SCENARIO_RUNNER_ROOT=C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner &&set PYTHONPATH=C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard;C:/Users/22780/Documents/CARLA_0.9.13/leaderboard/scenario_runner;C:/Users/22780/Documents/CARLA_0.9.10.1w/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg &&cd C:\Users\22780\Documents\CARLA_0.9.13\leaderboard\scenario_runner &&conda activate py37 &&python scenario_runner.py --route srunner/data/routes_debug.xml srunner/data/all_towns_traffic_scenarios1_3_4.json 0 --agent srunner/autoagents/myagent/npc_agent.py