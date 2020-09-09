# packages
import os
import sys
import json
import time
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
import heatmap.helpers     as hlp
import heatmap.miniclasses as mcl


class Director():
    """

    """

    def __init__(self, username, password, project_id, api_url_base, t_range=[0, 40], resolution=5, cache_dir='/tmp/', pickle_id='hmap_'):
        """
        Initialise Director class.

        Parameters
        ----------
        username : str
            DT Studio service account key.
        password : str
            DT Studio service account secret.
        project_id : str
            DT Studio project identifier.
        api_url_base : str
            Endpoint for API.
        t_range : [float, float]
            Temperature range [min, max] used in visualization.
        resolution : int
            Number of points per meter in heatmap grid.
        cache_dir : str
            Absolute path to directory used for caching distance maps.
        pickle_id : str
            Identifier used for files cached in cache_dir.

        """
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
        self.sample      = False
        self.cc = 0

        # set stream endpoint
        self.stream_endpoint = "{}/projects/{}/devices:stream".format(self.api_url_base, self.project_id)

        # parse system arguments
        self.__parse_sysargs()

        # set history- and streaming filters
        self.__set_filters()

        # inherit rooms layout
        self.__decode_json_layout()

        # get limits for x- and y- axes
        self.__generate_bounding_box()

        # generate distance map for each sensor
        if self.args['debug']:
            self.__eucledian_map_debug()
        else:
            self.__eucledian_map_threaded()

        # spawn heatmap
        self.heatmap = np.zeros(shape=self.X.shape)

        # check if sample is set
        if self.sample:
            print('\nUsing sample layout. No historic- of streaming data will be used.')
            print('For this, provide a layout using the --layout argument.')
            self.update_heatmap()
            self.plot_heatmap(show=True)
            sys.exit()


    def __parse_sysargs(self):
        """
        Parse for command line arguments.

        """

        # create parser object
        parser = argparse.ArgumentParser(description='Heatmap generation on Stream and Event History.')

        # get UTC time now
        now = (datetime.datetime.utcnow().replace(microsecond=0)).isoformat() + 'Z'

        # general arguments
        parser.add_argument('--layout',    metavar='', help='Json file with room layout.', required=False)
        parser.add_argument('--starttime', metavar='', help='Event history UTC starttime [YYYY-MM-DDTHH:MM:SSZ].', required=False, default=now)
        parser.add_argument('--endtime',   metavar='', help='Event history UTC endtime [YYYY-MM-DDTHH:MM:SSZ].',   required=False, default=now)
        parser.add_argument('--timestep',  metavar='', help='Heatmap update period.',      required=False, default=3600, type=int)

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
            for sensor in self.sensors + self.oofs:
                if source_id == sensor.sensor_id:
                    # give data to room
                    sensor.new_event_data(event_data)
                    if cout: print('-- New temperature {} for {} at [{}, {}].'.format(event_data['data']['temperature']['value'], source_id, sensor.x, sensor.y))
                    return True
        elif 'objectPresent' in event_data['data']:
            # find correct door
            for door in self.doors:
                if source_id == door.sensor_id:
                    # give state to door
                    door.new_event_data(event_data)
                    if cout: print('-- New door state {} for {} at [{}, {}].'.format(event_data['data']['objectPresent']['state'], source_id, door.x, door.y))
                    return True
        return False


    def __check_timestep(self, unixtime):
        """
        Check if more time than --timestep has passed since last heatmap update.

        Parameters
        ----------
        unixtime : int
            Seconds since 01-Jan 1970.

        Returns
        -------
        return : bool
            True if time to update heatmap.
            False if we're still waiting.

        """

        # check time since last update
        if self.last_update < 0:
            # update time to this event time
            self.last_update = unixtime
            return False

        elif unixtime - self.last_update > self.args['timestep']:
            # update timer to this event time
            self.last_update = unixtime

            return True


    def __decode_json_layout(self):
        """
        Parse json layout file and spawn related class objects.

        """

        # import json to dictionary
        if self.args['layout'] != None:
            path = self.args['layout']
        else:
            path = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'config', 'sample_layout.json')
            self.sample = True
        jdict = hlp.import_json(path)

        # count rooms and doors
        n_rooms = len(jdict['rooms'])
        n_doors = len(jdict['doors'])

        # initialise object lists
        self.rooms   = [mcl.Room() for i in range(n_rooms)]
        self.doors   = [mcl.Door() for i in range(n_doors)]

        # get rooms in dict
        for ri in range(n_rooms):
            # isolate room
            jdict_room = jdict['rooms'][ri]

            # count corners and sensors
            n_corners = len(jdict_room['corners'])
            n_sensors = len(jdict_room['sensors'])

            # adopt name
            self.rooms[ri].name = jdict_room['name']

            # give room list of corner and sensor objects
            self.rooms[ri].corners = [mcl.Corner(x=None, y=None) for i in range(n_corners)]
            self.rooms[ri].sensors = [mcl.Sensor(x=None, y=None) for i in range(n_sensors)]

            # update corners
            for ci in range(n_corners):
                # isolate json corner and give to room corner
                jdict_corner = jdict_room['corners'][ci]
                self.rooms[ri].corners[ci].set_coordinates(x=jdict_corner['x'], y=jdict_corner['y'])

            # update sensors
            for si in range(n_sensors):
                # isolate json sensor and give to room sensor
                jdict_sensor = jdict_room['sensors'][si]
                self.rooms[ri].sensors[si].post_initialise(jdict_sensor['x'], jdict_sensor['y'], jdict_sensor['sensor_id'], room_number=ri)

                # give t0 if exists
                if 't0' in jdict_sensor:
                    self.rooms[ri].sensors[si].t = jdict_sensor['t0']

        # get doors in dict
        for di in range(n_doors):
            # isolate doors
            jdict_door = jdict['doors'][di]

            # find rooms which door connects
            r1 = None
            r2 = None
            for room in self.rooms:
                if room.name == jdict_door['room1']:
                    r1 = room
                if room.name == jdict_door['room2']:
                    r2 = room

            # exit if rooms not found. Error in layout.
            if r1 == None or r2 == None:
                hlp.print_error('Error in layout. Door [{}] not connected to [{}] and [{}].'.format(jdict_door['name'], jdict_door['room1'], jdict_door['room2']), terminate=True)

            # reformat for easier updating
            p1 = [jdict_door['p1']['x'], jdict_door['p1']['y']]
            p2 = [jdict_door['p2']['x'], jdict_door['p2']['y']]

            # give variables to door object
            self.doors[di].post_initialise(p1, p2, r1, r2, jdict_door['sensor_id'], di)

            # give state if it exists
            if 'closed' in jdict_door:
                self.doors[di].closed = jdict_door['closed']


        # adopt all sensors to self
        self.sensors = []
        for room in self.rooms:
            for sensor in room.sensors:
                self.sensors.append(sensor)
        self.n_sensors = len(self.sensors)

        # get objects of interest in dict
        n_oofs    = len(jdict['oofs'])
        self.oofs = [mcl.Sensor(x=None, y=None) for i in range(n_oofs)]
        for i, oof in enumerate(jdict['oofs']):
            self.oofs[i].post_initialise(x=oof['x'], y=oof['y'], sensor_id=oof['sensor_id'], room_number=None)

            # give t0 if exists
            if 't0' in oof:
                self.oofs[i].t = oof['t0']


    def __generate_bounding_box(self):
        """
        Set grid dimension limits based on layout corners.

        """
        # find limits for x- and y-axis
        self.xlim = [0, 0]
        self.ylim = [0, 0]

        # iterate rooms
        for room in self.rooms:
            # iterate corners in room:
            for c in room.corners:
                if c.x < self.xlim[0]:
                    self.xlim[0] = c.x
                if c.x > self.xlim[1]:
                    self.xlim[1] = c.x
                if c.y < self.ylim[0]:
                    self.ylim[0] = c.y
                if c.y > self.ylim[1]:
                    self.ylim[1] = c.y

        # rounding
        self.xlim = [int(np.floor(self.xlim[0])), int(np.ceil(self.xlim[1]))]
        self.ylim = [int(np.floor(self.ylim[0])), int(np.ceil(self.ylim[1]))]

        # set maximum dimension for any axis
        self.maxdim = max(self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0])

        # generate interpolation axes
        self.x_interp = np.linspace(self.xlim[0], self.xlim[1], int(self.resolution*(self.xlim[1]-self.xlim[0])+0.5))
        self.y_interp = np.linspace(self.ylim[0], self.ylim[1], int(self.resolution*(self.ylim[1]-self.ylim[0])+0.5))

        # convert to compatible grid
        self.X, self.Y = np.meshgrid(self.x_interp, self.y_interp)


    def __populate_grid(self, D, N, M, corner, room):
        """
        Scan matrix and populate with eucledian distance for cells in line of sight of corner.

        Parameters
        ----------
        D : 2d ndarray
            Matrix to be populated.
        corner : object
            Corner Point object for which we check line of sight.

        Returns
        -------
        D : 2d ndarray
            Populated matrix.

        """

        # iterate x- and y-axis axis
        for x, gx in enumerate(self.x_interp):
            for y, gy in enumerate(self.y_interp):
                # set active node
                node = mcl.Point(self.x_interp[x], self.y_interp[y])

                # get distance from corner to node if in line of sight
                if not self.__has_direct_los(mcl.Point(corner.x+corner.dx, corner.y+corner.dy), node, room):
                    continue

                d = hlp.eucledian_distance(corner.x, corner.y, node.x, node.y)

                # update map if d is a valid value
                if d != None:

                    # add distance from sensor to corner
                    d += corner.dmin

                    # update map if less than existing value
                    if D[y, x] == 0 or d < D[y, x]:
                        D[y, x] = d
                        N[y, x] = len(corner.visited_doors)
                        M[y][x] = [door.number for door in corner.visited_doors]

        return D, N, M


    def __reset_pathfinding_variables(self):
        """
        Reset room, corner and door variables to their initial state.

        """

        for room in self.rooms:
            for corner in room.corners:
                corner.dmin = None
                corner.shortest_path = []
                corner.visited_doors = []
                corner.unused = True
        for door in self.doors:
            door.unused = True
            for of in [door.o1, door.o2]:
                of.dmin = None
                of.shortest_path = []
                of.visited_doors = []


    def __eucledian_map_debug(self):
        """
        Debug version of the eucledian distance mapping routine.
        Does the same as __eucledian_map_threaded(), but without multithreading.

        """

        # iterate sensors
        for i, sensor in enumerate(self.sensors):
            # initialise sensor distance map
            sensor.emap = np.zeros(shape=self.X.shape)

            # reset room corner distances
            self.__reset_pathfinding_variables()

            # recursively find shortest distance to all valid corners
            path  = []
            doors = []
            _, _ = self.__find_shortest_paths(sensor, self.rooms[sensor.room_number], path, doors, dr=0)

            # initialise grids
            sensor.D = np.zeros(shape=self.X.shape)
            sensor.N = np.zeros(shape=self.X.shape)
            sensor.M = [[[] for y in range(self.X.shape[1])] for x in range(self.X.shape[0])]

            # populate map from sensor poitn of view
            sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, sensor, self.rooms[sensor.room_number])
            if 1:
                self.plot_debug(start=sensor, grid=[sensor.D])

            # populate grid with distances from each corner
            for ri, room in enumerate(self.rooms):
                # fill from doors
                for di, door in enumerate(self.doors):
                    print('Sensor {}, Room {}, Door {}'.format(i, ri, di))
                    if door.outbound_room == room:
                        offset_node = door.outbound_offset
                        if len(offset_node.shortest_path) > 0:
                            sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, offset_node, room)

                            # plot population process
                            if 1:
                                self.plot_debug(start=sensor, grid=[sensor.D], paths=offset_node.shortest_path)

                # fill from corners
                for ci, corner in enumerate(room.corners):
                    print('Sensor {}, Room {}, Corner {}'.format(i, ri, ci))
                    if len(corner.shortest_path) > 0:
                        sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, corner, room)

                        # plot population process
                        if 1:
                            self.plot_debug(start=sensor, grid=[sensor.D], paths=corner.shortest_path)

            # plot population result
            if 1:
                self.plot_debug(start=sensor, grid=[sensor.D])


    def __eucledian_map_threaded(self):
        """
        Generate eucledian distance map for each sensor.
        Applies multiprocessing for a significant reduction in execution time.

        """

        def map_process(sensor, i):
            """
            Same as __eucledian_map_threaded() but must be isolated in a function for multiprocessing.
            Writes populated distance maps to cache_dir so that we only have to do this once. It's slow.

            Parameters
            ----------
            sensor : object
                Sensor object with coordinates and temperature information.
            i : int
                Sensor number in list.

            """

            self.__reset_pathfinding_variables()
        
            # recursively find shortest path from sensor to all corners
            path  = []
            doors = []
            _, _ = self.__find_shortest_paths(sensor, self.rooms[sensor.room_number], path, doors, dr=0)
        
            # initialise grids
            sensor.D = np.zeros(shape=self.X.shape)
            sensor.N = np.zeros(shape=self.X.shape)
            sensor.M = [[[] for y in range(self.X.shape[1])] for x in range(self.X.shape[0])]

            # populate map from sensor poitn of view
            sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, sensor, self.rooms[sensor.room_number])
        
            # populate grid with distances from each corner
            for ri, room in enumerate(self.rooms):
                # fill from doors
                for di, door in enumerate(self.doors):
                    print('Mapping Grid for Sensor {:>3}, Room {:>3},   Door {:>3}'.format(i, ri, di))
                    if door.outbound_room == room:
                        offset_node = door.outbound_offset
                        if len(offset_node.shortest_path) > 0:
                            sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, offset_node, room)

                # fill from corners
                for ci, corner in enumerate(room.corners):
                    print('Mapping Grid for Sensor {:>3}, Room {:>3}, Corner {:>3}'.format(i, ri, ci))
                    if len(corner.shortest_path) > 0:
                        sensor.D, sensor.N, sensor.M = self.__populate_grid(sensor.D, sensor.N, sensor.M, corner, room)

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
        """
        Exchange self.sensors with sensors cached in cache_dir.
        Usually called to recover previously calculated distance maps.

        """

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
                    pickle_path = os.path.join(self.cache_dir, self.pickle_id + '{}.pkl'.format(i))
                    pickle_sensor = hlp.read_pickle(pickle_path, cout=True)

                    # exchange
                    self.sensors[i].D = pickle_sensor.D
                    self.sensors[i].N = pickle_sensor.N
                    self.sensors[i].M = pickle_sensor.M

                    # found it
                    found = True

            # shouldn't happen, but just in case
            if not found:
                hlp.print_error('Pickle at [{}] does not exist. Try running without --read.'.format(pickle_path), terminate=True)


    def __find_shortest_paths(self, start, room, path, doors, dr):
        """
        Recursively find the shortest path from sensor to every corner in layout.

        Parameters
        ----------
        start : object
            Point object of were we currently have point of view.
        room : object
            Room object of which room we are currently in.
        path : list
            List of previously visited points in the current recursive branch.
        doors : list
            List of doors which have been passed through in the current recursive branch.
        dr : float
            Total distance traveled from initial sensor start location.

        Returns
        -------
        path : list
            List of visited points in the current recursive branch, including current.
        doors : list
            List of doors which have been passed through in the current recursive branch.

        """

        # append path with active node
        path.append(start)

        # stop if we've been here before on a shorter path
        if start.dmin != None and dr > start.dmin:
            return path, doors
        
        # as this is currently the sortest path from sensor to active, copy it to active
        start.dmin = dr
        start.shortest_path = [p for p in path]
        start.visited_doors = [d for d in doors]

        # find candidate corners for path expansion
        corner_candidates = self.__get_corner_candidates(start, room)
        door_candidates   = self.__get_door_candidates(start, room)

        # plot candidates
        if 0:
            self.plot_debug(start=start, goals=corner_candidates + door_candidates, show=False)

        # recursively iterate candidates
        for c in corner_candidates:
            # calculate distance to candidate
            ddr = hlp.eucledian_distance(start.x, start.y, c.x, c.y)

            # recursive
            path, doors = self.__find_shortest_paths(c, room, path, doors, dr+ddr)
            path.pop()
        for c in corner_candidates:
            c.unused = True

        for d in door_candidates:
            # calculate distance to candidate
            ddr = hlp.eucledian_distance(start.x, start.y, d.inbound_offset.x, d.inbound_offset.y)

            # fix offset
            d.outbound_offset.dx = 0
            d.outbound_offset.dy = 0

            # append to doors list
            doors.append(d)

            # recursive
            path, doors = self.__find_shortest_paths(d.outbound_offset, d.outbound_room, path, doors, dr+ddr)

            # pop lists as we're back to current depth
            path.pop()
            doors.pop()

        for d in door_candidates:
            d.unused = True

        return path, doors


    def __get_corner_candidates(self, start, room):
        """
        Return a list of corners which can be used as next step in recursive __find_shortest_paths().

        Parameters
        ----------
        start : object
            Point object of were we currently have point of view.
        room : object
            Room object of which room we are currently in.

        Returns
        -------
        candidates : list
            List of corners in room which can be used for next recursive step.

        """

        # initialise list
        candidates = []

        # iterate corners in room
        for i, corner in enumerate(room.corners):
            # skip visisted
            if not corner.unused:
                continue

            # get offset
            dx, dy = self.__corner_offset(room.corners, i)

            # check if corner is candidate material
            if self.__has_direct_los(mcl.Point(start.x+start.dx, start.y+start.dy), mcl.Point(corner.x+dx, corner.y+dy), room):
                corner.dx = dx
                corner.dy = dy
                candidates.append(corner)
                corner.unused = False

        return candidates


    def __get_door_candidates(self, start, room):
        """
        Return a list of doors which can be passed through as next step in recursive __find_shortest_paths().

        Parameters
        ----------
        start : object
            Point object of were we currently have point of view.
        room : object
            Room object of which room we are currently in.

        Returns
        -------
        candidates : list
            List of doors in room which can be passed through.

        """

        # initialise list
        candidates = []

        # iterate corners in room
        for door in self.doors:
            # skip visisted
            if not door.unused:
                continue

            # check if we have LOS to either offset
            offset_start = mcl.Point(start.x+start.dx, start.y+start.dy)
            if self.__has_direct_los(offset_start, door.o1, room):
                if room == door.room1:
                    door.outbound_room = door.room2
                else:
                    door.outbound_room = door.room1
                door.inbound_offset  = door.o1
                door.outbound_offset = door.o2
                candidates.append(door)
                door.unused = False
            elif self.__has_direct_los(offset_start, door.o2, room):
                if room == door.room1:
                    door.outbound_room = door.room2
                else:
                    door.outbound_room = door.room1
                door.inbound_offset  = door.o2
                door.outbound_offset = door.o1
                candidates.append(door)
                door.unused = False

        return candidates


    def __has_direct_los(self, start, goal, room):
        """
        Check if start has line of sight (LOS) to goal.

        Parameters
        ----------
        start : object
            Point object used as point of view.
        goal : object
            Point object we check if we have LOS to.

        Returns
        -------
        return : float
            Returns eucledian distance from start to goal if LOS is True.
            Returns None if no LOS.

        """

        # check if los
        for i in range(len(room.corners)):
            # two corners define a wall which can be intersected
            ir = i + 1
            if ir > len(room.corners)-1:
                ir = 0

            if self.__line_intersects(start, goal, room.corners[i], room.corners[ir]):
                return False
        
        return True


    def __line_intersects(self, p1, q1, p2, q2): 
        """
        Determine if two lines intersect in 2-D space.

        Parameters
        ----------
        p1 : float
            x-coordinate of first line.
        q1 : float
            y-coordinate of first line.
        p2 : float
            x-coordinate of second line.
        q2 : float
            y-coordinate of second line.

        Returns
        -------
        return : bool
            True if lines intersect.
            False if no intersect.

        """

        # find the 4 orientations required for the general and special cases 
        o1 = self.__orientation(p1, q1, p2) 
        o2 = self.__orientation(p1, q1, q2) 
        o3 = self.__orientation(p2, q2, p1) 
        o4 = self.__orientation(p2, q2, q1) 
      
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # special Cases 
      
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
      
        # if none of the cases 
        return False


    def __orientation(self, p, q, r): 
        """
        Find the orientation of an ordered triplet (p,q,r) function.
        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ for details.

        Parameters
        ----------
        p : float
            First triplet index.
        q : float
            Second triplet index.
        r : float
            Third triplet index.

        Returns
        -------
        return : int
            0 if colinear points 
            1 if clockwise points 
            2 if counterclockwise 

        """
          
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
        """
        Determine if q is on the segment p-r.

        Parameters
        ----------
        p : float
            First triplet index.
        q : float
            Second triplet index.
        r : float
            Third triplet index.

        Returns
        -------
        return : bool
            True if on segment.
            False if not on segment.

        """

        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
               (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False


    def __corner_offset(self, corners, i, eps=1/1e3):
        """
        Generate a tiny offset in corner convex direction.

        Parameters
        ----------
        corners : list
            List of corner objects in a room.
        i : int
            Index of current corner of interest in corner list.
        eps : float
            Distance of offset. Should be small.

        Returns
        -------
        x_offset : float
            Offset in the x-direction.
        y_offset : float
            Offset in the y-direction.

        """

        # circular buffer behavior for list edges
        il = i - 1
        if il < 0:
            il = -1
        ir = i + 1
        if ir > len(corners) - 1:
            ir = 0

        # isolate corner triplet around corner of interest
        pl = corners[il]
        pc = corners[i]
        pr = corners[ir]

        # get complex direction of corner triplet
        mx = np.sign(((pc.x - pl.x) + (pc.x - pr.x)) / 2)
        my = np.sign(((pc.y - pl.y) + (pc.y - pr.y)) / 2)

        # plot for debugging purposes
        if 0:
            plt.cla()
            for room in self.rooms:
                xx, yy = room.get_outline()
                plt.plot(xx, yy, '-k', linewidth=3)
            plt.plot(pl.x, pl.y, 'or')
            plt.plot(pr.x, pr.y, 'og')
            plt.plot(pc.x, pc.y, 'ok')
            plt.plot([pc.x, pl.x], [pc.y, pl.y], 'o-r', linewidth=3)
            plt.plot([pc.x, pr.x], [pc.y, pr.y], 'o-g', linewidth=3)
            plt.plot([pc.x, pc.x+mx], [pc.y, pc.y+my], 'o--k')
            plt.waitforbuttonpress()

        # multiply by epsilon
        x_offset = mx * eps
        y_offset = my * eps

        return x_offset, y_offset


    def update_heatmap(self):
        """
        Using calculated distance- and door maps, update heatmap with temperature data.

        """

        # iterate x- and y-axis axis
        for x, gx in enumerate(self.x_interp):
            for y, gy in enumerate(self.y_interp):
                # reset lists
                temperatures = []
                distances    = []
                weights      = []

                # iterate sensors
                for room in self.rooms:
                    for sensor in room.sensors:
                        los = True
                        # check if doors in path are closed
                        if len(sensor.M[y][x]) > 0:
                            for door in self.doors:
                                if door.closed and door.number in sensor.M[y][x]:
                                    los = False

                        # check if distance grid is valid here
                        if los and sensor.D[y, x] > 0 and sensor.t != None:
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
                

    def __set_filters(self):
        """
        Set filters for data fetched through API.

        """

        # historic events
        self.history_params = {
            'page_size': 1000,
            'start_time': self.args['starttime'],
            'end_time': self.args['endtime'],
            'event_types': ['temperature', 'objectPresent']
        }

        # stream events
        self.stream_params = {
            'event_types': ['temperature', 'objectPresent']
        }


    def __fetch_event_history(self):
        """
        For each sensor in project, request all events since --starttime from API.

        """

        # initialise empty event list
        self.event_history = []

        # combine temperature- and door sensors
        project_sensors = [s for s in self.sensors] + [d for d in self.doors] + [o for o in self.oofs]

        # iterate devices
        for sensor in project_sensors:
            # isolate id
            sensor_id = sensor.sensor_id

            # skip sensors without id
            if sensor_id is None:
                continue

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
        
                if self.history_params['page_token'] != '':
                    print('\t-- paging')
        
        # sort event history in time
        self.event_history.sort(key=hlp.json_sort_key, reverse=False)


    def __initialise_stream_temperatures(self):
        """
        When initialising a stream, set initial sensor temperature values as last value reported.

        """

        # get list of sensors in project
        device_list_url = "{}/projects/{}/devices".format(self.api_url_base, self.project_id)

        # request
        device_listing = requests.get(device_list_url, auth=(self.username, self.password)).json()
        for device in device_listing['devices']:
            name = os.path.basename(device['name'])

            if 'temperature' in device['reported']:
                for sensor in self.sensors + self.oofs:
                    if name == sensor.sensor_id:
                        sensor.t = device['reported']['temperature']['value']
            elif 'objectPresent' in device['reported']:
                for door in self.doors:
                    if name == door.sensor_id:
                        state = device['reported']['objectPresent']['state']
                        if state == 'PRESENT':
                            door.closed = True
                        else:
                            door.closed = False


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
            if 'temperature' in event_data['data']:
                update_time = event_data['data']['temperature']['updateTime']
            else:
                update_time = event_data['data']['objectPresent']['updateTime']
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

        # do nothing if sample
        if self.sample:
            return

        # if no history were used, get last events from sensors
        if not self.fetch_history:
            self.__initialise_stream_temperatures()

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
                    event_data = json.loads(event.data)

                    # check for event data error
                    if 'error' in event_data:
                        hlp.print_error('Event with error msg [{}]. Skipping Event.'.format(event_data['error']['message']), terminate=False)

                    # new data received
                    event_data = event_data['result']['event']
        
                    # serve event to director
                    served = self.__new_event_data(event_data, cout=True)

                    # plot progress
                    if served and self.args['plot']:
                        # get event time in unixtime
                        if 'temperature' in event_data['data']:
                            update_time = event_data['data']['temperature']['updateTime']
                        else:
                            update_time = event_data['data']['objectPresent']['updateTime']
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
            except KeyError:
                print('Skipping event due to KeyError.')
                print(event_data)
                print()
            
            # wait 1s before attempting to reconnect
            time.sleep(1)


    def initialise_debug_plot(self):
        self.fig, self.ax = plt.subplots()


    def plot_debug(self, start=None, goals=None, grid=None, paths=None, show=False):
        # initialise if if not open
        if not hasattr(self, 'ax') or not plt.fignum_exists(self.fig.number):
            self.initialise_debug_plot()

        # clear
        self.ax.clear()

        # draw walls
        for room in self.rooms:
            xx, yy = room.get_outline()
            self.ax.plot(xx, yy, '-k', linewidth=3)

        # draw doors
        for door in self.doors:
            self.ax.plot(door.xx, door.yy, 'g', linewidth=5)

        # draw goal node
        if goals != None and start != None:
            for g in goals:
                self.ax.plot([start.x, g.x], [start.y, g.y], '.-r', markersize=10)

        # draw start node
        if start != None:
            self.ax.plot(start.x, start.y, 'ok', markersize=10)

        # draw paths
        if paths != None:
            for i in range(len(paths)-1):
                p1 = paths[i]
                p2 = paths[i+1]
                self.ax.plot([p1.x, p2.x], [p1.y, p2.y], '.-r')

        # plot grid
        if grid != None:
            for g in grid:
                pc = self.ax.contourf(self.X.T, self.Y.T, g.T, max(1, int(g.max()-g.min())))
                pc.set_clim(0, max(self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0]))

        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()
        else:
            plt.waitforbuttonpress()


    def initialise_construction_plot(self):
        self.cfig, self.cax = plt.subplots()
        self.cfig.set_figheight(self.ylim[1]-self.ylim[0])
        self.cfig.set_figwidth(self.xlim[1]-self.xlim[0])
        self.cax.set_xlim([self.xlim[0]-1, self.xlim[1]+1])
        self.cax.set_ylim([self.ylim[0]-1, self.ylim[1]+1])
        # self.cax.axis('off')


    def plot_construction(self, terminate=False):
        # initialise if not open
        if not hasattr(self, 'cax') or not plt.fignum_exists(self.cfig.number):
            self.initialise_construction_plot()

        for room in self.rooms:
            for corner in room.corners:
                self.cax.plot(corner.x, corner.y, 'ok')
            plt.waitforbuttonpress()
            xx, yy = room.get_outline()
            for i in range(len(xx)-1):
                self.cax.plot(xx[i:i+2], yy[i:i+2], '-k', linewidth=3)
                plt.waitforbuttonpress()
            for sensor in room.sensors:
                self.cax.plot(sensor.x, sensor.y, 'ok', markersize=10)
                plt.waitforbuttonpress()


        if terminate:
            sys.exit()


    def initialise_heatmap_plot(self):
        self.hfig, self.hax = plt.subplots()
        self.hfig.set_figheight(self.ylim[1]-self.ylim[0])
        self.hfig.set_figwidth(self.xlim[1]-self.xlim[0])
        self.hfig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=self.t_range[0], vmax=self.t_range[1]), cmap=cm.jet))


    def plot_heatmap(self, update_time='', blocking=True, show=False):
        # initialise if not open
        if not hasattr(self, 'hax') or not plt.fignum_exists(self.hfig.number):
            self.initialise_heatmap_plot()

        # clear
        self.hax.clear()

        # set title
        self.hax.set_title(update_time)

        # draw walls
        for room in self.rooms:
            xx, yy = room.get_outline()
            self.hax.plot(xx, yy, '-k', linewidth=3)

        # draw doors
        for door in self.doors:
            self.hax.plot(door.xx, door.yy, '-k', linewidth=14)
            if door.closed:
                self.hax.plot(door.xx, door.yy, '-', color='orangered', linewidth=8)
            else:
                self.hax.plot(door.xx, door.yy, '-', color='limegreen', linewidth=8)

        # draw sensors
        for sensor in self.sensors:
            self.hax.plot(sensor.x, sensor.y, 'xk', markersize=10, markeredgewidth=2.5)

        # draw heatmap
        pc = self.hax.contourf(self.X.T, self.Y.T, self.heatmap.T, (self.t_range[1]-self.t_range[0])*5, cmap=cm.jet)
        # pc = self.hax.contourf(self.X.T, self.Y.T, self.heatmap.T, 100, cmap=cm.jet)
        pc.set_clim(self.t_range[0], self.t_range[1])

        # draw oofs
        # for oof in self.oofs:
        #     if oof.t is not None:
        #         t = (oof.t-self.t_range[0])/(self.t_range[1]-self.t_range[0])
        #         self.hax.plot(oof.x, oof.y, 'o', color=pc.cmap(t), markeredgecolor='k', markersize=20)

        # lock aspect
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.tight_layout()
        # plt.xlabel('Distance [m]')
        # plt.ylabel('Distance [m]')

        if 0:
            self.hfig.savefig('/home/kepler/tmp/' + '{:09d}.png'.format(self.cc), dpi=self.hfig.dpi, bbox_inches='tight')
            self.cc += 1
        else:
            if blocking:
                if show:
                    plt.show()
                else:
                    plt.waitforbuttonpress()
            else:
                plt.pause(0.01)

