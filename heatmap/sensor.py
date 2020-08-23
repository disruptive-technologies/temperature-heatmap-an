# project
import heatmap.helpers as hlp


class Sensor():

    def __init__(self, sensor):
        # unpack sensor parameters
        self.__unpack_sensor_params(sensor)

        # variables
        self.t = None


    def __unpack_sensor_params(self, sensor):
        # id
        self.id = sensor['id']

        # coordinates
        if 'x' in sensor.keys() and 'y' in sensor.keys():
            self.x  = sensor['x']
            self.y  = sensor['y']
        else:
            hlp.print_error('Sensors must be initialised with x- and y-coordinates.')

        # initial temperature
        if 't0' in sensor.keys():
            self.t0 = sensor['t0']
        else:
            self.t0 = None

