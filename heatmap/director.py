# packages
import os
import sys
import requests
import datetime
import argparse
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib          import cm
from matplotlib.path     import Path
from matplotlib.colors   import Normalize
from matplotlib.patches  import PathPatch

# project
import heatmap.helpers   as hlp
import config.parameters as prm
from heatmap.room import Room


class Director():
    """

    """

    def __init__(self, username, password, project_id, api_url_base):
        # give to self
        self.username     = username
        self.password     = password
        self.project_id   = project_id
        self.api_url_base = api_url_base

        # variables
        self.last_update = -1
        self.cc = 0

        # set stream endpoint
        self.stream_endpoint = "{}/projects/{}/devices:stream".format(self.api_url_base, self.project_id)

        # parse system arguments
        self.__parse_sysargs()

        # spawn room instances
        self.__spawn_rooms()

        # get devices in project
        self.__fetch_project_devices()

        # set filters for fetching data
        self.__set_filters()


    def __parse_sysargs(self):
        """
        Parse for command line arguments.

        """

        # create parser object
        parser = argparse.ArgumentParser(description='Desk Occupancy Estimation on Stream and Event History.')

        # get UTC time now
        now = (datetime.datetime.utcnow().replace(microsecond=0)).isoformat() + 'Z'

        # general arguments
        parser.add_argument('--starttime', metavar='', help='Event history UTC starttime [YYYY-MM-DDTHH:MM:SSZ].', required=False, default=now)
        parser.add_argument('--endtime',   metavar='', help='Event history UTC endtime [YYYY-MM-DDTHH:MM:SSZ].',   required=False, default=now)
        parser.add_argument('--timestep',  metavar='', help='Heatmap update period.', required=False, default=3600, type=int)

        # boolean flags
        parser.add_argument('--plot',   action='store_true', help='Plot the estimated desk occupancy.')

        # convert to dictionary
        self.args = vars(parser.parse_args())

        # set history flag
        if now == self.args['starttime']:
            self.fetch_history = False
        else:
            self.fetch_history = True



        # set filters for fetching data
        self.__set_filters()


    def __set_filters(self):
        """
        Set filters for data fetched through API.

        """

        # historic events
        self.history_params = {
            'page_size': 1000,
            'start_time': self.args['starttime'],
            'end_time': self.args['endtime'],
            'event_types': ['temperature']
        }

        # stream events
        self.stream_params = {
            'event_types': ['temperature']
        }


    def __spawn_rooms(self):
        """
        Spawn one Room instance per room in config.

        """

        # initialise list of rooms
        self.rooms = []

        # iterate rooms in config
        for room in prm.rooms:
            # append instance to list
            self.rooms.append(Room(room))

        # set bounding box
        self.__set_dimensions()
    

    def __set_dimensions(self):
        xax = [None, None]
        yax = [None, None]
        for room in prm.rooms:
            for i in range(len(room['corners']['x'])):
                x = room['corners']['x'][i]
                y = room['corners']['y'][i]
                if xax[0] is None or x < xax[0]:
                    xax[0] = x
                if xax[1] is None or x > xax[1]:
                    xax[1] = x
                if yax[0] is None or y < yax[0]:
                    yax[0] = y
                if yax[1] is None or y > yax[1]:
                    yax[1] = y

        self.width = xax[1] - xax[0]
        self.height = yax[1] - yax[0]
        # print('w: {},  h: {}'.format(self.width, self.height))


    def __fetch_project_devices(self):
        """
        Fetch information about all devices in project.

        """

        # request list
        devices_list_url = "{}/projects/{}/devices".format(self.api_url_base,  self.project_id)
        device_listing = requests.get(devices_list_url, auth=(self.username, self.password))
        
        # check error code
        if device_listing.status_code < 300:
            # remove fluff
            devices = device_listing.json()['devices']

            # isolate id
            project_sensors = [os.path.basename(device['name']) for device in devices]
        else:
            print(device_listing.json())
            hlp.print_error('Status Code: {}'.format(device_listing.status_code), terminate=True)

        # give devices in both project and rooms to self
        self.sensors = []

        # iterate rooms
        for room in self.rooms:
            # iterate sensors in room
            for sensor in room.sensors:
                # iterate sensors in project
                if sensor.id in project_sensors:
                    # save if in both parameters and project
                    self.sensors.append(sensor.id)


    def __fetch_event_history(self):
        """
        For each sensor in project, request all events since --starttime from API.

        """

        # initialise empty event list
        self.event_history = []

        # iterate devices
        for sensor_id in self.sensors:
            # some printing
            print('-- Getting event history for {}'.format(sensor_id))
        
            # initialise next page token
            self.history_params['page_token'] = None
        
            # set endpoints for event history
            event_list_url = "{}/projects/{}/devices/{}/events".format(self.api_url_base, self.project_id, sensor_id)
        
            # perform paging
            while self.history_params['page_token'] != '':
                event_listing = requests.get(event_list_url, auth=(self.username, self.password), params=self.history_params)
                event_json = event_listing.json()

                if event_listing.status_code < 300:
                    self.history_params['page_token'] = event_json['nextPageToken']
                    self.event_history += event_json['events']
                else:
                    print(event_json)
                    hlp.print_error('Status Code: {}'.format(event_listing.status_code), terminate=True)
        
                if self.history_params['page_token'] is not '':
                    print('\t-- paging')
        
        # sort event history in time
        self.event_history.sort(key=hlp.json_sort_key, reverse=False)


    def __update_heatmap(self):
        # iterate rooms
        for room in self.rooms:
            # update room-specific map
            room.update_heatmap()


    def __new_event_data(self, event_data, cout=True):
        """
        Receive new event_data json and pass it along to the correct room instance.

        Parameters
        ----------
        event_data : dictionary
            Data json containing new event data.
        cout : bool
            Will print event information to console if True.

        """

        # get id of source sensor
        source_id = os.path.basename(event_data['targetName'])

        # verify temperature event
        if 'temperature' in event_data['data'].keys():
            # iterate rooms
            for room in self.rooms:
                # skip rooms with no sensors
                if len(room.sensors) == 0:
                    continue
                
                # check if sensor is in this room
                for sensor in room.sensors:
                    if source_id == sensor.id:
                        # give data to room
                        room.new_event_data(source_id, event_data)

            # get event time in unixtime
            update_time = event_data['data']['temperature']['updateTime']
            _, unixtime = hlp.convert_event_data_timestamp(update_time)

            # check timer
            if self.last_update < 0:
                # update time to this event time
                self.last_update = unixtime
            elif unixtime - self.last_update > self.args['timestep']:
                # update timer to this event time
                self.last_update = unixtime

                # plot if set
                if self.args['plot']:
                    # update heatmap
                    self.__update_heatmap()

                    # show plot
                    self.plot(update_time, blocking=True)


    def run_history(self):
        """
        Iterate through and calculate occupancy for event history.

        """

        # do nothing if starttime not given
        if not self.fetch_history:
            return

        # get list of hsitoric events
        self.__fetch_event_history()

        # initialise plot before loop
        if self.args['plot']:
            self.initialise_plot()
        
        # estimate occupancy for history 
        cc = 0
        for i, event_data in enumerate(self.event_history):
            cc = hlp.loop_progress(cc, i, len(self.event_history), 25, name='event history')
            # serve event to director
            self.__new_event_data(event_data, cout=False)
        

    def initialise_plot(self):
        self.pfig, self.pax = plt.subplots(dpi=100)
        self.pfig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=prm.temperature_range[0], vmax=prm.temperature_range[1]), cmap=cm.jet))


    def plot(self, update_time, blocking=False):
        self.pax.clear()
        # iterate rooms
        for r in self.rooms:
            # get room wall outline
            outline = r.get_outline()

            # set map
            cmap = cm.jet

            # contour
            # pc = self.pax.contourf(r.X, r.Y, r.Z, prm.temperature_range[1]-prm.temperature_range[0], cmap=cmap)
            pc = self.pax.contourf(r.X, r.Y, r.Z, 100, cmap=cmap)

            # scatter
            # sc = self.pax.scatter(r.x, r.y, 250, r.z, cmap=cm.jet, edgecolors='k')

            # set temperature scale
            if r.Z is not None:
                pc.set_clim(prm.temperature_range[0], prm.temperature_range[1])
                # sc.set_clim(prm.temperature_range[0], prm.temperature_range[1])

            # draw outline
            self.pax.plot(outline[0], outline[1], '-k', linewidth=2)
            for door in r.doors:
                self.pax.plot(door['x'], door['y'], '-k', linewidth=10)

            # mask
            clippath = Path(np.c_[r.corners['x'], r.corners['y']])
            patch = PathPatch(clippath, facecolor='none')
            self.pax.add_patch(patch)
            for c in pc.collections:
                c.set_clip_path(patch)

            sc = []
            for i in range(len(r.x)):
                n = (r.z[i]-prm.temperature_range[0]) / (prm.temperature_range[1]-prm.temperature_range[0])
                c = plt.Circle((r.x[i], r.y[i]), 0.25, color='k')
                self.pax.add_artist(c)
                c.set_clip_path(patch)

        # blocking
        if blocking:
            plt.title('Blocking @ t={}'.format(update_time))

            if 0:
                self.pfig.set_figheight(8)
                self.pfig.set_figwidth(30)
                out = '/home/kepler/tmp/'
                self.pfig.savefig(out + '{:09d}.png'.format(self.cc), dpi=100, bbox_inches='tight')
                self.cc += 1
            else:
                # scale axes equally
                plt.gca().set_aspect('equal', adjustable='box')
                plt.waitforbuttonpress()
                # plt.show()

