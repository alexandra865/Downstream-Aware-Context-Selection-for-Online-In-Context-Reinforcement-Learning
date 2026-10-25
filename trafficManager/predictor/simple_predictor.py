'''
Author: Licheng Wen
Date: 2022-12-02 14:41:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''

import math

import numpy as np
from common.observation import Observation
from predictor.abstract_predictor import AbstractPredictor, Prediction
from trafficManager.common.vehicle import VehicleType

from utils.roadgraph import RoadGraph
from utils.trajectory import State, Trajectory


class UncontrolledPredictor(AbstractPredictor):

    @staticmethod
    def _cartesian_constant_velocity_prediction(vehicle, config):
        """Predict in Cartesian space when the route graph cannot cover the horizon."""
        predict_t = config["MIN_T"]
        dt = config["DT"]
        state = vehicle.current_state
        speed = float(state.vel if state.vel is not None else state.s_d)
        yaw = float(state.yaw) if state.yaw is not None else 0.0
        x = float(state.x)
        y = float(state.y)

        predict_trajectory = Trajectory()
        for t in np.arange(0, predict_t, dt):
            predict_trajectory.states.append(
                State(
                    t=t,
                    x=x + speed * math.cos(yaw) * t,
                    y=y + speed * math.sin(yaw) * t,
                    yaw=yaw,
                    vel=speed,
                    acc=0.0,
                    d=state.d,
                    s=state.s + speed * t,
                    s_d=speed,
                    laneID=vehicle.lane_id,
                )
            )
        return predict_trajectory.states

    @staticmethod
    def _lane_chain(vehicle, roadgraph, required_s):
        lane = roadgraph.get_lane_by_id(vehicle.lane_id)
        if lane is None or lane.course_spline is None:
            return []

        lanes = [lane]
        visited = {lane.id}
        available_s = float(lane.course_spline.s[-1])

        while available_s < required_s:
            next_lane = roadgraph.get_available_next_lane(
                lanes[-1].id,
                vehicle.available_lanes,
            )
            if next_lane is None:
                next_lane = roadgraph.get_next_lane(lanes[-1].id)
            if (
                next_lane is None
                or next_lane.id in visited
                or next_lane.course_spline is None
            ):
                break

            lanes.append(next_lane)
            visited.add(next_lane.id)
            available_s += max(float(next_lane.course_spline.s[-1]) - 0.1, 0.0)

        return lanes

    @classmethod
    def _constant_velocity_prediction(cls, vehicle, roadgraph, config):
        """Rebuild a finite prediction when a cached trajectory is exhausted."""
        predict_t = config["MIN_T"]
        dt = config["DT"]
        state = vehicle.current_state
        speed = float(state.s_d if state.s_d is not None else state.vel)
        required_s = float(state.s) + max(speed, 0.0) * float(predict_t)
        lanes = cls._lane_chain(vehicle, roadgraph, required_s)

        if not lanes:
            return cls._cartesian_constant_velocity_prediction(vehicle, config)

        predict_trajectory = Trajectory()
        s = float(state.s)
        d = float(state.d)
        for t in np.arange(0, predict_t, dt):
            predict_trajectory.states.append(
                State(t=t, d=d, s=s, s_d=speed, laneID=vehicle.lane_id)
            )
            s += speed * dt

        try:
            predict_trajectory.frenet_to_cartesian(lanes, state)
        except (ArithmeticError, IndexError, TypeError, ValueError, Warning):
            return cls._cartesian_constant_velocity_prediction(vehicle, config)

        if not predict_trajectory.states:
            return cls._cartesian_constant_velocity_prediction(vehicle, config)
        return predict_trajectory.states

    def predict(
        self, observation: Observation, roadgraph: RoadGraph,
        lastseen_vehicles, through_timestep, config) -> Prediction:
        prediction = Prediction()

        for vehicle in observation.vehicles:
            if vehicle.vtype != VehicleType.OUT_OF_AOI:
                remaining_prediction = []
                if vehicle.id in lastseen_vehicles:
                    remaining_prediction = lastseen_vehicles[
                        vehicle.id
                    ].trajectory.states[through_timestep:]
                if remaining_prediction:
                    prediction.results[vehicle] = remaining_prediction
                else:
                    prediction.results[vehicle] = (
                        self._constant_velocity_prediction(
                            vehicle,
                            roadgraph,
                            config,
                        )
                    )
            else:
                prediction.results[vehicle] = self._constant_velocity_prediction(
                    vehicle,
                    roadgraph,
                    config,
                )
        return prediction
