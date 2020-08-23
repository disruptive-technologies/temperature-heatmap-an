# packages
import sys
import numpy as np

# project
import heatmap.helpers   as hlp
import config.parameters as prm
from heatmap.sensor import Sensor


class Room():
    """

    """

    def __init__(self, room):
        # unpack parameters
        self.__unpack_room_params(room)

        # generate bounding box and interpolation meshgrid
        self.__generate_meshgrid()

        # spawn sensor object for each temperature sensor in room
        self.__spawn_sensors()


    def __unpack_room_params(self, room):
        # corner coordinates
        if 'corners' in room.keys():
            self.corners = room['corners']
        else:
            hlp.print_error('Room can not be initialised without corner coordinates.')

        # doors
        if 'doors' in room.keys():
            self.doors = room['doors']
        else:
            self.doors = []

        # sensors
        if 'sensors' in room.keys():
            self.sensor_dicts = room['sensors']
        else:
            self.sensor_dicts = []


    def __generate_meshgrid(self):
        # get limits
        self.xlim = [min(self.corners['x']), max(self.corners['x'])]
        self.ylim = [min(self.corners['y']), max(self.corners['y'])]

        # generate interpolation axes
        self.x_interp = np.linspace(self.xlim[0], self.xlim[1], int(prm.resolution*(self.xlim[1]-self.xlim[0])+0.5))
        self.y_interp = np.linspace(self.ylim[0], self.ylim[1], int(prm.resolution*(self.ylim[1]-self.ylim[0])+0.5))

        # convert to compatible grid
        self.X, self.Y = np.meshgrid(self.x_interp, self.y_interp)


    def __spawn_sensors(self):
        # initialise sensor list
        self.sensors = []

        # stop if no sensors in room
        if len(self.sensor_dicts) == 0:
            return

        # iterate sensors in room
        for sensor in self.sensor_dicts:
            # verify identifier exists
            if 'id' not in sensor.keys():
                hlp.print_error('Sensor should not be provided without an ID.')

            # append sensor instance
            self.sensors.append(Sensor(sensor))


    def __inverse_distance_weighted(self, p=2):
        self.Z = np.zeros(shape=self.X.shape)

        # iterate x-axis
        for x, xv in enumerate(self.x_interp):
            # iterate y-axis
            for y, yv in enumerate(self.y_interp):
                # iterate sensors
                temperatures = []
                distances    = []
                for sensor in self.sensors:
                    # skip no temperature
                    if sensor.t is None:
                        continue

                    # calculate distance to sensor
                    distances.append(hlp.eucledian_distance([sensor.x, sensor.y], [xv, yv]))
                    temperatures.append(sensor.t)

                # check for 0 distance to avoid nan
                if 0 in distances:
                    self.Z[y, x] = temperatures[np.where(np.array(distances)==0)[0][0]]
                else:
                    # weighted average
                    weights = (1/(np.array(distances)))**p
                    temperatures = np.array(temperatures)

                    # update mesh
                    self.Z[y, x] = sum(weights*temperatures) / sum(weights)


    def get_outline(self):
        return self.corners['x'] + [self.corners['x'][0]], self.corners['y'] + [self.corners['y'][0]]


    def new_event_data(self, sensor_id, event):
        # update temperature for sensor
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                sensor.t = event['data']['temperature']['value']


    def update_heatmap(self):
        # collect x, y and z
        self.x = np.array([sensor.x for sensor in self.sensors])
        self.y = np.array([sensor.y for sensor in self.sensors])
        self.z = np.array([sensor.t for sensor in self.sensors])

        # no sensors gives blank output
        if len(self.sensors) == 0:
            self.Z = np.zeros(shape=self.X.shape)*np.nan
        # single sensor gives no gradient
        elif len(self.sensors) == 1:
            self.Z = np.ones(shape=self.X.shape)*self.sensors[0].t
        # interpolation for multiple sensors
        else:
            # run inverse distance algorithmn
            self.__inverse_distance_weighted(p=2)
