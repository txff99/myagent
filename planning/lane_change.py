import carla    


def change_lane(waypoint):
    """
    This method is in charge of overtaking behaviors.
        :param location: current location of the agent
        :param waypoint: current waypoint of the agent
        :param vehicle_list: list of all the nearby vehicles
    """

    left_turn = waypoint.left_lane_marking.lane_change
    right_turn = waypoint.right_lane_marking.lane_change

    left_wpt = waypoint.get_left_lane()
    right_wpt = waypoint.get_right_lane()

    if (left_turn == carla.LaneChange.Left or left_turn ==
        carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
      
        print("Overtaking to the left!")
        return left_wpt.transform.location
                                    
    elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
        
        print("Overtaking to the right!")
        return right_wpt.transform.location
    else:
        print("Cannot change")
        return None