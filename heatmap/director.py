# packages
import os
import sys
import json
import requests
import datetime
import argparse
import sseclient
import numpy           as np
import multiprocessing as mpr

# plotting
import matplotlib.pyplot as plt
from matplotlib          import cm
from matplotlib.colors   import Normalize

# project
import heatmap.helpers   as hlp
import config.layout     as layout


class Director():
    """

    """

    def __init__(self, username, password, project_id, api_url_base, t_range=[0, 40], resolution=5, cache_dir='/tmp/', pickle_id='hmap_'):
        # give to self
        self.username     = username
        self.password     = password
        self.project_id   = project_id
        self.api_url_base = api_url_base
        self.t_range      = t_range
        self.resolution   = resolution
        self.cache_dir    = cache_dir
        self.pickle_id    = pickle_id

        # variables
        self.last_update = -1
        self.cc = 0

        # set stream endpoint
        self.stream_endpoint = "{}/projects/{}/devices:stream".format(self.api_url_base, self.project_id)

        # parse system arguments
        self.__parse_sysargs()

        # adopt layout
        self.corners = layout.corners
        self.walls   = layout.walls
        self.sensors = layout.sensors

        # get sensors in project
        self.__fetch_project_sensors()

        # set filters for fetching data
        self.__set_filters()

        # get limits for x- and y- axes
        self.__set_bounding_box()

        # generate meshgrid
        self.__generate_meshgrid()

        # spawn heatmap
        self.heatmap = np.zeros(shape=self.X.shape)

        # pre-calculate sensor distances in grid
        if self.args['debug']:
            self.__eucledian_map_debug()
        else:
            self.__eucledian_map_threaded()


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
        parser.add_argument('--plot',  action='store_true', help='Plot the estimated desk occupancy.')
        parser.add_argument('--debug', action='store_true', help='Disables multithreading for debug visualization.')
        parser.add_argument('--read',  action='store_true', help='Import cached distance maps.')

        # convert to dictionary
        self.args = vars(parser.parse_args())

        # set history flag
        if now == self.args['starttime']:
            self.fetch_history = False
        else:
            self.fetch_history = True



        # set filters for fetching data
        self.__set_filters()


    def __set_bounding_box(self):
        # find limits for x- and y-axis
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        for c in self.corners:
            if c.x < self.xlim[0]:
                self.xlim[0] = c.x
            if c.x > self.xlim[1]:
                self.xlim[1] = c.x
            if c.y < self.ylim[0]:
                self.ylim[0] = c.y
            if c.y > self.ylim[1]:
                self.ylim[1] = c.y
        self.xlim = [int(np.floor(self.xlim[0])), int(np.ceil(self.xlim[1]))]
        self.ylim = [int(np.floor(self.ylim[0])), int(np.ceil(self.ylim[1]))]

        # set maximum dimension for any axis
        self.maxdim = max(self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0])


    def __generate_meshgrid(self):
        # generate interpolation axes
        self.x_interp = np.linspace(self.xlim[0], self.xlim[1], int(self.resolution*(self.xlim[1]-self.xlim[0])+0.5))
        self.y_interp = np.linspace(self.ylim[0], self.ylim[1], int(self.resolution*(self.ylim[1]-self.ylim[0])+0.5))

        # convert to compatible grid
        self.X, self.Y = np.meshgrid(self.x_interp, self.y_interp)


    def __eucledian_map_debug(self):
        # iterate sensors
        for i, sensor in enumerate(self.sensors):
            # reset corner distances
            for corner in self.corners:
                corner.shortest_distance = None
                corner.shortest_path     = []

            # recursively find shortest path from sensor to all corners
            path = []
            self.__find_shortest_paths(sensor.p, path, dr=0)

            # initialise grid
            sensor.D = np.zeros(shape=self.X.shape)

            # populate map from sensor poitn of view
            sensor.D = self.__populate_grid(sensor.D, sensor.p)

            # plot population process
            if 0:
                self.plot_debug(start=sensor.p, grid=[sensor.D])

            # populate grid with distances from each corner
            for ci, corner in enumerate(self.corners):
                if len(corner.shortest_path) > 0:
                    print('sensor {}, corner {}'.format(i, ci))
                    sensor.D = self.__populate_grid(sensor.D, corner)

                    # plot population process
                    if 0:
                        self.plot_debug(goal=corner, grid=[sensor.D], paths=[corner.shortest_path])

            # plot population result
            if 1:
                self.plot_debug(start=sensor.p, grid=[sensor.D])


    def __eucledian_map_threaded(self):
        def map_process(sensor, i):
            # reset corner distances
            for corner in self.corners:
                corner.shortest_distance = None
                corner.shortest_path     = []
        
            # recursively find shortest path from sensor to all corners
            path = []
            self.__find_shortest_paths(sensor.p, path, dr=0)
        
            # initialise grid
            sensor.D = np.zeros(shape=self.X.shape)
        
            # populate map from sensor poitn of view
            sensor.D = self.__populate_grid(sensor.D, sensor.p)
        
            # populate grid with distances from each corner
            for ci, corner in enumerate(self.corners):
                if len(corner.shortest_path) > 0:
                    print('Populating map for sensor {} at corner {}.'.format(i, ci))
                    sensor.D = self.__populate_grid(sensor.D, corner)

            # write sensor object to pickle
            hlp.write_pickle(sensor, os.path.join(self.cache_dir, self.pickle_id + '{}.pkl'.format(i)), cout=True)

        # just skip everything and read from cache if so desired
        if self.args['read']:
            self.__get_cached_sensors()
            return

        # initialise variables needed for process
        procs = []
        nth_proc = 0

        # iterate sensors
        for i, sensor in enumerate(self.sensors):
            # spawn a thread per sensor
            proc = mpr.Process(target=map_process, args=(sensor, i))
            procs.append(proc)
            proc.start()
            print('-- Process #{} spawned.'.format(nth_proc))
            nth_proc = nth_proc + 1

        # wait for each individual process to finish
        nth_proc = 0
        for proc in procs:
            proc.join()
            print('-- Process #{} completed.'.format(nth_proc))
            nth_proc = nth_proc + 1

        # fetch sensors from cache
        self.__get_cached_sensors()


    def __get_cached_sensors(self):
        # get files in cache
        cache_files = os.listdir(self.cache_dir)

        # iterate sensors
        for i in range(self.n_sensors):
            # keep track of if we found the pickle
            found = False

            # iterate files in cache
            for f in cache_files:
                # look for correct pickle
                if self.pickle_id + '{}.pkl'.format(i) in f and not found:
                    # read pickle
                    pickle_sensor = hlp.read_pickle(os.path.join(self.cache_dir, self.pickle_id + '{}.pkl'.format(i)), cout=True)

                    # exchange
                    # sensors.append(pickle_sensor)
                    self.sensors[i].D = pickle_sensor.D

                    # found it
                    found = True

            # shouldn't happen, but just in case
            if not found:
                hlp.print_error('Pickle #{} has gone missing.'.format(i), terminate=True)


    def __populate_grid(self, D, corner):
        # iterate x- and y-axis axis
        for x, gx in enumerate(self.x_interp):
            for y, gy in enumerate(self.y_interp):
                # set active node
                node = hlp.Point(self.x_interp[x], self.y_interp[y])

                # get distance from corner to node if in line of sight
                d = self.__pathfind(corner, node)

                # update map if d is a valid value
                if d != None:
                    # add distance from sensor to corner
                    d += corner.shortest_distance

                    # update map if less than existing value
                    if D[y, x] == 0 or d < D[y, x]:
                        D[y, x] = d

        return D


    def __pathfind(self, start, goal):
        # draw a straight line
        straight = hlp.Line(start, goal)

        # check if los
        los = True
        for wall in self.walls:
            if start not in wall.pp:
                if self.__line_intersects(start, goal, wall.p1, wall.p2):
                    los = False
                # concave check
                elif len(start.walls) == 2 and self.__is_concave(goal, start):
                    los = False
        if los:
            return hlp.eucledian_distance(start.x, start.y, goal.x, goal.y)
        return None


    def __find_shortest_paths(self, active, path, dr):
        # append path with active node
        path.append([active.x, active.y])

        # stop if we've been here before on a shorter path
        if active.shortest_distance != None and dr > active.shortest_distance:
            return path
        
        # as this is currently the sortest path from sensor to active, copy it to active
        active.shortest_distance = dr
        active.shortest_path = [p for p in path]

        # path search plot
        if 0:
            self.plot(start=active, paths=[path] + [c.shortest_path for c in corners])

        # find candidate corners for path expansion
        candidates = []

        # nudge in convex directions
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            # create offset point
            m = 1000
            offset = hlp.Point(active.x+dx/m, active.y+dy/m)

            # validate nudge is convex
            if not self.__is_concave(offset, active):
                for corner in self.corners:
                    # skip used
                    if corner.unused:
                        # validate corner
                        ddr = hlp.eucledian_distance(active.x, active.y, corner.x, corner.y)
                        if corner.shortest_distance == None or dr + ddr < corner.shortest_distance:
                            if self.__validate_corner(offset, corner):
                                candidates.append(corner)
                                corner.unused = False

        if 0:
            # plot
            self.plot(start=active, candidates=candidates, paths=[path])

        # recursively iterate candidates
        for c in candidates:
            # calculate distance to candidate
            ddr = hlp.eucledian_distance(active.x, active.y, c.x, c.y)

            # recursive
            path = self.__find_shortest_paths(c, path, dr+ddr)
            path.pop()
        for c in candidates:
            c.unused = True

        return path


    def __validate_corner(self, origin, corner):
        # skip if more than 2 branches
        if len(corner.walls) > 2 or len(corner.walls) == 0:
            return False
        
        # skip if concave
        if len(corner.walls) == 2 and self.__is_concave(origin, corner):
            return False

        # skip if corner is relative to origin
        if len(origin.walls) == 2 and self.__is_concave(corner, origin):
            return False

        # skip is no los
        for wall in self.walls:
            if corner not in wall.pp and origin not in wall.pp:
                if self.__line_intersects(origin, corner, wall.p1, wall.p2):
                    return False
        
        # passed
        return True


    def __orientation(self, p, q, r): 
        # to find the orientation of an ordered triplet (p,q,r) 
        # function returns the following values: 
        # 0 : Colinear points 
        # 1 : Clockwise points 
        # 2 : Counterclockwise 
          
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
        # for details of below formula.  
          
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
        if (val > 0): 
              
            # Clockwise orientation 
            return 1
        elif (val < 0): 
              
            # Counterclockwise orientation 
            return 2
        else: 
              
            # Colinear orientation 
            return 0


    def __on_segment(self, p, q, r): 
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
               (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False


    def __line_intersects(self, p1,q1,p2,q2): 
        # Find the 4 orientations required for  
        # the general and special cases 
        o1 = self.__orientation(p1, q1, p2) 
        o2 = self.__orientation(p1, q1, q2) 
        o3 = self.__orientation(p2, q2, p1) 
        o4 = self.__orientation(p2, q2, q1) 
      
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # Special Cases 
      
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
        if ((o1 == 0) and self.__on_segment(p1, p2, q1)): 
            return True
      
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
        if ((o2 == 0) and self.__on_segment(p1, q2, q1)): 
            return True
      
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
        if ((o3 == 0) and self.__on_segment(p2, p1, q2)): 
            return True
      
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
        if ((o4 == 0) and self.__on_segment(p2, q1, q2)): 
            return True
      
        # If none of the cases 
        return False


    def __is_concave(self, node, point):
        # a point with a single wall connected (endpoint) is always convex
        if len(point.walls) < 2:
            return False

        # corner to check
        p0 = point
        x0 = p0.x
        y0 = p0.y
    
        # define where we are
        xx = 1
        yy = 1
        if node.x < p0.x:
            xx = -1
        if node.y < p0.y:
            yy = -1
    
        wall = point.walls[0]
        for p in wall.pp:
            if p != point:
                p1 = p
        wall = point.walls[1]
        for p in wall.pp:
            if p != point:
                p2 = p

        # stop if straight line
        if abs(p1.x - p2.x) == 0 or abs(p1.y - p2.y) == 0:
            return True

        tx = (p1.x + p2.x) / 2
        ty = (p1.y + p2.y) / 2
    
        txx = 1
        tyy = 1
        if tx < p0.x:
            txx = -1
        if ty < p0.y:
            tyy = -1
    
        if txx == xx and tyy == yy:
            bounded = True
        else:
            bounded = False
    
        if 0:
            print(bounded)
            plt.plot(p0.x, p0.y, 'or', markersize=15)
            plt.plot(p1.x, p1.y, 'og', markersize=15)
            plt.plot(p2.x, p2.y, 'og', markersize=15)
            plt.plot(tx, ty, '*b', markersize=15)
            plt.axvline(y0)
            plt.axhline(x0)
            plt.plot(node.x, node.y, '*k', markersize=25)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.waitforbuttonpress()
    
        return bounded


    def __fetch_project_sensors(self):
        """
        Fetch information about sensors in project from API.

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
        validated_sensors = []

        # iterate rooms
        for sensor in self.sensors:
            # iterate sensors in room
            if sensor.identifier in project_sensors:
                # save if in both parameters and project
                validated_sensors.append(sensor)
            else:
                print('Sensor [{}] were not found in project. Skipping...'.format(sensor.identifier))

        # keep only sensors both in project and layout
        self.sensors   = validated_sensors
        self.n_sensors = len(self.sensors)


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


    def __fetch_event_history(self):
        """
        For each sensor in project, request all events since --starttime from API.

        """

        # initialise empty event list
        self.event_history = []

        # iterate devices
        for sensor in self.sensors:
            # isolate id
            sensor_id = sensor.identifier

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
            # check if sensor is in this room
            for sensor in self.sensors:
                if source_id == sensor.identifier:
                    # give data to room
                    sensor.new_event_data(event_data)
                    if cout: print('-- New Event for {}.'.format(source_id))
                    return True
        return False


    def __check_timestep(self, unixtime):
        # check time since last update
        if self.last_update < 0:
            # update time to this event time
            self.last_update = unixtime
            return False

        elif unixtime - self.last_update > self.args['timestep']:
            # update timer to this event time
            self.last_update = unixtime

            return True


    def update_heatmap(self):
        # iterate x- and y-axis axis
        for x, gx in enumerate(self.x_interp):
            for y, gy in enumerate(self.y_interp):
                # reset lists
                temperatures = []
                distances    = []

                # iterate sensors
                for sensor in self.sensors:
                    # check if distance grid is valid here
                    if sensor.D[y, x] > 0 and sensor.t != None:
                        temperatures.append(sensor.t)
                        distances.append(sensor.D[y, x])

                # do nothing if no valid distances
                if len(distances) == 0:
                    self.heatmap[y, x] = None
                elif len(distances) == 1:
                    self.heatmap[y, x] = temperatures[0]
                else:
                    # calculate weighted average
                    weights = (1/(np.array(distances)))**2
                    temperatures = np.array(temperatures)
                    
                    # update mesh
                    self.heatmap[y, x] = sum(weights*temperatures) / sum(weights)


    def run_history(self):
        """
        Iterate through and calculate occupancy for event history.

        """

        # do nothing if starttime not given
        if not self.fetch_history:
            return

        # get list of hsitoric events
        self.__fetch_event_history()
        
        # estimate occupancy for history 
        cc = 0
        for i, event_data in enumerate(self.event_history):
            cc = hlp.loop_progress(cc, i, len(self.event_history), 25, name='event history')
            # serve event to director
            _ = self.__new_event_data(event_data, cout=False)

            # get event time in unixtime
            update_time = event_data['data']['temperature']['updateTime']
            _, unixtime = hlp.convert_event_data_timestamp(update_time)

            # plot if timestep has passed
            if self.__check_timestep(unixtime):
                # update heatmap
                self.update_heatmap()

                # plot
                self.plot_heatmap(update_time=update_time, blocking=True)


    def run_stream(self, n_reconnects=5):
        """
        Update heatmap for realtime stream data from sensors.

        Parameters
        ----------
        n_reconnects : int
            Number of reconnection attempts at disconnect.

        """

        # cout
        print("Listening for events... (press CTRL-C to abort)")

    
        # initial plot
        if self.args['plot']:
            # initialise heatmap
            self.update_heatmap()

            # plot
            self.plot_heatmap(update_time='t0', blocking=False)
    
        # loop indefinetly
        nth_reconnect = 0
        while nth_reconnect < n_reconnects:
            try:
                # reset reconnect counter
                nth_reconnect = 0
        
                # get response
                response = requests.get(self.stream_endpoint, auth=(self.username, self.password), headers={'accept':'text/event-stream'}, stream=True, params=self.stream_params)
                client = sseclient.SSEClient(response)
        
                # listen for events
                print('Connected.')
                for event in client.events():
                    # new data received
                    event_data = json.loads(event.data)['result']['event']
        
                    # serve event to director
                    served = self.__new_event_data(event_data, cout=True)

                    # plot progress
                    if served and self.args['plot']:
                        # get event time in unixtime
                        update_time = event_data['data']['temperature']['updateTime']
                        _, unixtime = hlp.convert_event_data_timestamp(update_time)

                        # update heatmap
                        self.update_heatmap()

                        # Plot
                        self.plot_heatmap(update_time=update_time, blocking=False)
            
            # catch errors
            # Note: Some VPNs seem to cause quite a lot of packet corruption (?)
            except requests.exceptions.ConnectionError:
                nth_reconnect += 1
                print('Connection lost, reconnection attempt {}/{}'.format(nth_reconnect, n_reconnects))
            except requests.exceptions.ChunkedEncodingError:
                nth_reconnect += 1
                print('An error occured, reconnection attempt {}/{}'.format(nth_reconnect, n_reconnects))
            
            # wait 1s before attempting to reconnect
            time.sleep(1)


    def initialise_debug_plot(self):
        self.fig, self.ax = plt.subplots()


    def plot_debug(self, start=None, goal=None, lines=None, grid=None, candidates=None, paths=None):
        # initialise if if not open
        if not hasattr(self, 'ax') or not plt.fignum_exists(self.fig.number):
            self.initialise_debug_plot()

        # clear
        self.ax.clear()

        # draw walls
        for wall in self.walls:
            self.ax.plot(wall.xx, wall.yy, '-k', linewidth=3)

        # draw lines
        if lines != None:
            for line in lines:
                self.ax.plot(line.xx, line.yy, '-r')

        # draw candidate corners
        if candidates != None:
            for c in candidates:
                self.ax.plot(c.x, c.y, 'ob', markersize=10)

        # draw active node
        if goal != None:
            self.ax.plot(goal.x, goal.y, 'or', markersize=10)

        # draw active sensor
        if start != None:
            self.ax.plot(start.x, start.y, 'or', markersize=10)

        # plot grid
        if grid != None:
            for g in grid:
                pc = self.ax.contourf(self.X.T, self.Y.T, g.T, max(1, int(g.max()-g.min())))
                pc.set_clim(0, max(self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0]))

        # plot path
        if paths != None:
            for path in paths:
                for i in range(1, len(path)):
                    xx = [path[i-1][0], path[i][0]]
                    yy = [path[i-1][1], path[i][1]]
                    self.ax.plot(xx, yy, '.-r')

        if 0:
            self.fig.set_figheight(self.ylim[1] - self.ylim[0])
            self.fig.set_figwidth(self.xlim[1] - self.xlim[0])
            out = '/home/kepler/tmp/'
            self.fig.savefig(out + '{:09d}.png'.format(self.cc), dpi=100, bbox_inches='tight')
            self.cc += 1
        else:
            plt.gca().set_aspect('equal', adjustable='box')
            plt.waitforbuttonpress()


    def initialise_heatmap_plot(self):
        self.hfig, self.hax = plt.subplots()
        self.hfig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=self.t_range[0], vmax=self.t_range[1]), cmap=cm.jet))


    def plot_heatmap(self, update_time='', blocking=True):
        # initialise if not open
        if not hasattr(self, 'hax') or not plt.fignum_exists(self.hfig.number):
            self.initialise_heatmap_plot()

        # clear
        self.hax.clear()

        # set title
        self.hax.set_title(update_time)

        # draw walls
        for wall in self.walls:
            self.hax.plot(wall.xx, wall.yy, '-k', linewidth=3)

        # draw sensors
        for sensor in self.sensors:
            self.hax.plot(sensor.p.x, sensor.p.y, 'ok', markersize=10)

        # draw heatmap
        pc = self.hax.contourf(self.X.T, self.Y.T, self.heatmap.T, self.t_range[1]-self.t_range[0], cmap=cm.jet)
        # pc = self.hax.contourf(self.X.T, self.Y.T, self.heatmap.T, 100, cmap=cm.jet)
        pc.set_clim(self.t_range[0], self.t_range[1])

        if 0:
            self.hfig.set_figheight(self.ylim[1] - self.ylim[0])
            self.hfig.set_figwidth(self.xlim[1] - self.xlim[0])
            out = '/home/kepler/tmp/'
            self.hfig.savefig(out + '{:09d}.png'.format(self.cc), dpi=100, bbox_inches='tight')
            self.cc += 1
        else:
            # lock aspect
            plt.gca().set_aspect('equal', adjustable='box')

            if blocking:
                plt.waitforbuttonpress()
            else:
                plt.pause(0.01)

