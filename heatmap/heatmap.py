import json
import random
import pickle
from datetime import datetime
from pathlib import Path
from multiprocessing import Process

import numpy as np
import disruptive as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

matplotlib.use('TkAgg')


PICKLE_PREFIX = 'hmap_'
T_RANGE = [15, 30]


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def write_pickle(obj, path: str):
    with open(path, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_pickle(path: str):
    with open(path, 'rb') as input:
        return pickle.load(input)


class Point():
    def __init__(self, x: int, y: int):
        # Cartesian coordinates of the point.
        self.x: int = x
        self.y: int = y
        self.dx: int = 0  # offset in x-direction
        self.dy: int = 0  # offset in y-direction


class Corner(Point):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y)

        self.unused = True


class Sensor(Point):
    def __init__(self,
                 x: int,
                 y: int,
                 device_id: str,
                 sensor_number: int,
                 ) -> None:
        super().__init__(x, y)
        self.device_id: str = device_id
        self.sensor_number: int = sensor_number
        self.device: dt.Device | None = self._fetch_api_object()

    @property
    def project_id(self) -> str:
        if self.device is None:
            return ''
        else:
            return self.device.project_id

    def _valid_sensor_id(self) -> bool:
        if len(self.device_id) == 20:
            return True
        elif len(self.device_id) == 23 and self.device_id.startswith('emu'):
            return True
        else:
            return False

    def _fetch_api_object(self) -> dt.Device | None:
        if self._valid_sensor_id():
            return dt.Device.get_device(self.device_id)
        else:
            return None


class TemperatureSensor(Sensor):
    def __init__(self,
                 x: int,
                 y: int,
                 device_id: str,
                 sensor_number: int,
                 room_number: int,
                 ) -> None:
        super().__init__(x, y, device_id, sensor_number)
        self.room_number: int = room_number
        self.temperature: float = self._init_temperature()

        self.dmin = None

    def _init_temperature(self) -> bool:
        # If `device` is None, i.e. we couldn't find any device with
        # the provided `sensor_id` in the API, use random value in range.
        if self.device is None:
            return random.randint(21, 28)
        else:
            if self.device.reported.temperature is not None:
                return self.device.reported.temperature.celsius
            elif self.device.reported.humidity is not None:
                return self.device.reported.humidity.celsius
            else:
                raise ValueError('No temperature or humidity reported.')

    def update(self, event):
        assert event.device_id == self.device.device_id
        self.temperature = event.data.celsius


class Room():
    def __init__(self,
                 name: str,
                 corners: list[Corner],
                 temperature_sensors: list[TemperatureSensor],
                 ) -> None:
        self.name: str = name
        self.corners: list[Corner] = corners
        self.sensors: list[TemperatureSensor] = temperature_sensors

    def get_outline(self):
        """
        Returns lists of x- and y-coordinates for walls in room.
        Walls are defined as straight lines between adjecent corners.
        Therefore, the first index must be appended to end for complete outline.

        Returns
        -------
        xax : list
            List of x-coordinates of corners in room with single index overflow.
        yax : list
            List of y-coordinates of corners in room with single index overflow.

        """

        xax = [c.x for c in self.corners] + [self.corners[0].x]
        yax = [c.y for c in self.corners] + [self.corners[0].y]

        return xax, yax


class Line():
    """
    Class to represent a line in 2-D space.
    Defined by two Point class objects.

    """

    def __init__(self, p1, p2):
        """
        Line class constructor.

        Parameters
        ----------
        p1 : object
            Point object of line start.
        p2 : object
            Point object of line end.

        """

        # give to self
        self.point1 = p1
        self.point2 = p2

        # some variable formatting for easier access
        self.pp = [p1, p2]
        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


class Door(Line, Sensor):
    def __init__(self,
                 point1: Point,
                 point2: Point,
                 room1: Room,
                 room2: Room,
                 device_id: str,
                 door_number: int
                 ) -> None:
        Line.__init__(self, point1, point2)
        Sensor.__init__(
            self,
            x=(point1.x + point2.x) // 2,
            y=(point1.y + point2.y) // 2,
            device_id=device_id,
            sensor_number=door_number,
        )

        self.room1: Room = room1
        self.room2: Room = room2
        self.door_number: int = door_number

        # Create perpendicular bisector on door line.
        self.o1, self.o2 = self._perpendicular_bisector(eps=1/1e3)

        self.state = self._init_state()
        self.closed = self.state == dt.events.ObjectPresent.STATE_PRESENT

    def _perpendicular_bisector(self, eps):
        v = [self.point2.y - self.point1.y, self.point2.x - self.point1.x]
        return (
            Corner(self.x+v[0]*(+eps), self.y+v[1]*(+eps)),
            Corner(self.x+v[0]*(-eps), self.y+v[1]*(-eps)),
        )

    def _init_state(self) -> bool:
        # If `device` is None, i.e. we couldn't find any device with
        # the provided `sensor_id` in the API, assume "open" state.
        if self.device is None:
            return dt.events.ObjectPresent.STATE_NOT_PRESENT
        else:
            return self.device.reported.object_present.state

    def update(self, event):
        assert event.device_id == self.device.device_id
        self.state = event.data.state
        self.closed = self.state == dt.events.ObjectPresent.STATE_PRESENT


class Heatmap():
    def __init__(self,
                 layout_path: str = '',
                 threaded: bool = False,
                 debug: bool = False,
                 ) -> None:
        self._abs_dir: Path = Path(__file__).parent.absolute()

        if threaded and debug:
            raise ValueError('Cannot run threaded and debug simultaneously.')

        self.debug: bool = debug

        # A lot of stuff depends on the layout configuration file, like the 
        # heatmap array size, bounding box, etc. Therefore, we
        # will do all compilation in the constructor directly.
        self._decode_layout_json(layout_path)
        self._generate_bounding_box()
        self._euclidean_map(threaded=threaded)

        # Empty array for temperature heatmap data.
        # Heatmap is filled by calling `update_heatmap()`.
        self.heatmap = np.zeros(shape=self.X.shape)

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
                                if door.closed and door.door_number in sensor.M[y][x]:
                                    los = False

                        # check if distance grid is valid here
                        if los and sensor.D[y, x] > 0 and sensor.temperature != None:
                            temperatures.append(sensor.temperature)
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

    def _decode_layout_json(self, layout_path: str):
        # If `layout_path` is empty string, use provided example layout.
        if len(layout_path) == 0:
            layout_path = self._abs_dir / '../config' / 'example_layout.json'
        else:
            layout_path = Path(layout_path)

        if not layout_path.exists():
            raise FileNotFoundError(f'Layout file not found: {layout_path}')

        layout_dict: dict | None = {}
        with open(layout_path) as f:
            layout_dict: dict = json.load(f)
        assert isinstance(layout_dict, dict), 'Error loading JSON layout.'

        # Iterate and initialize each room in the layout.
        rooms: list[Room] = []
        for room_index in range(len(layout_dict['rooms'])):
            room_dict = layout_dict['rooms'][room_index]

            # Check if the last point is the same as the first point.
            # If not, add it to the end. This is important when calculating
            # line of sight as rooms needs to be completley "closed".
            if room_dict['corners'][0] != room_dict['corners'][-1]:
                room_dict['corners'].append(room_dict['corners'][0])

            # Iterate and initialize each corner in the room.
            corners: list[Corner] = [
                Corner(c['x'], c['y']) for c in room_dict['corners']
            ]
            sensors: list[TemperatureSensor] = [TemperatureSensor(
                x=s['x'],
                y=s['y'],
                device_id=s['sensor_id'],
                room_number=room_index,
                sensor_number=i,
            ) for i, s in enumerate(room_dict['sensors'])]

            rooms.append(Room(
                name=room_dict['name'],
                corners=corners,
                temperature_sensors=sensors,
            ))

        # Iterate and initialize each door in the layout.
        doors: list[Door] = []
        for door_index in range(len(layout_dict['doors'])):
            door_dict = layout_dict['doors'][door_index]

            # Find the two rooms that the door connects.
            room1: Room | None = None
            room2: Room | None = None
            for room in rooms:
                if room.name == door_dict['room1']:
                    room1: Room = room
                if room.name == door_dict['room2']:
                    room2: Room = room

            assert isinstance(room1, Room), f'Door {door_index} missing Room 1'
            assert isinstance(room2, Room), f'Door {door_index} missing Room 2'

            doors.append(Door(
                point1=Point(
                    x=door_dict['p1']['x'],
                    y=door_dict['p1']['y'],
                ),
                point2=Point(
                    x=door_dict['p2']['x'],
                    y=door_dict['p2']['y'],
                ),
                room1=room1,
                room2=room2,
                device_id=door_dict['sensor_id'],
                door_number=door_index,
            ))

        self.rooms: list[Room] = rooms
        self.doors: list[Door] = doors

    def _generate_bounding_box(self):
        xlim: list[int] = [0, 0]
        ylim: list[int] = [0, 0]

        for room in self.rooms:
            for corner in room.corners:
                if corner.x < xlim[0]:
                    xlim[0] = corner.x
                if corner.x > xlim[1]:
                    xlim[1] = corner.x
                if corner.y < ylim[0]:
                    ylim[0] = corner.y
                if corner.y > ylim[1]:
                    ylim[1] = corner.y

        # Round to nearest "outer" integer.
        xlim = [int(np.floor(xlim[0])), int(np.ceil(xlim[1]))]
        ylim = [int(np.floor(ylim[0])), int(np.ceil(ylim[1]))]

        # Maximum possible dimension is the largest of the x- and y-axes.
        self.maxdim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

        # Generate interpolation axes with some resolution.
        res = 5
        self.x_interp = np.linspace(
            start=xlim[0],
            stop=xlim[1],
            num=int(res * (xlim[1] - xlim[0]) + 0.5),
        )
        self.y_interp = np.linspace(
            start=ylim[0],
            stop=ylim[1],
            num=int(res * (ylim[1] - ylim[0]) + 0.5),
        )

        # Convert to compatible grid.
        self.X, self.Y = np.meshgrid(self.x_interp, self.y_interp)

        self.xlim = xlim
        self.ylim = ylim

    def _reset_pathfinding_variables(self):
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

    def _get_corner_candidates(self, start):
        """
        Return a list of corners which can be used as next
        step in recursive _find_shortest_paths().

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
        for room in self.rooms:
            for i, corner in enumerate(room.corners):
                # skip visisted
                if not corner.unused:
                    continue

                # get offset
                dx, dy = self._corner_offset(room.corners, i)

                # check if corner is candidate material
                if self._has_direct_los(Point(start.x+start.dx, start.y+start.dy), Point(corner.x+dx, corner.y+dy)):
                    corner.dx = dx
                    corner.dy = dy
                    candidates.append(corner)
                    corner.unused = False

        return candidates

    def _get_door_candidates(self, start, room):
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
            offset_start = Point(start.x+start.dx, start.y+start.dy)
            if self._has_direct_los(offset_start, door.o1):
                if room == door.room1:
                    door.outbound_room = door.room2
                else:
                    door.outbound_room = door.room1
                door.inbound_offset = door.o1
                door.outbound_offset = door.o2
                candidates.append(door)
                door.unused = False
            elif self._has_direct_los(offset_start, door.o2):
                if room == door.room1:
                    door.outbound_room = door.room2
                else:
                    door.outbound_room = door.room1
                door.inbound_offset = door.o2
                door.outbound_offset = door.o1
                candidates.append(door)
                door.unused = False

        return candidates

    def _corner_offset(self, corners, i, eps=1/1e3):
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

        # multiply by epsilon
        x_offset = mx * eps
        y_offset = my * eps

        return x_offset, y_offset

    def _has_direct_los(self, start, goal):
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
        return : bool
            Has line of sight or not.

        """

        # check if los in room
        for room in self.rooms:
            for i in range(len(room.corners) - 1):
                # two corners define a wall which can be intersected
                if self._line_intersects(start, goal, room.corners[i], room.corners[i+1]):
                    return False

        return True

    def _line_intersects(self, p1, q1, p2, q2): 
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
        o1 = self._orientation(p1, q1, p2) 
        o2 = self._orientation(p1, q1, q2) 
        o3 = self._orientation(p2, q2, p1) 
        o4 = self._orientation(p2, q2, q1) 
      
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # special Cases 
      
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
        if ((o1 == 0) and self._on_segment(p1, p2, q1)): 
            return True
      
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
        if ((o2 == 0) and self._on_segment(p1, q2, q1)): 
            return True
      
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
        if ((o3 == 0) and self._on_segment(p2, p1, q2)): 
            return True
      
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
        if ((o4 == 0) and self._on_segment(p2, q1, q2)): 
            return True
      
        # if none of the cases 
        return False

    def _orientation(self, p, q, r): 
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

    def _on_segment(self, p, q, r): 
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

    def _find_shortest_paths(self, start, room, path, doors, dr):

        # Append active node to path.
        path.append(start)

        # Stop if we've been here before on a shorter path.
        if start.dmin is not None and dr > start.dmin:
            return path, doors

        # Copy to active as current sortest path from sensor to active.
        start.dmin = dr
        start.shortest_path = [p for p in path]
        start.visited_doors = [d for d in doors]

        # Find candidate corners for path expansion.
        corner_candidates = self._get_corner_candidates(start)
        door_candidates = self._get_door_candidates(start, room)

        # recursively iterate candidates
        for c in corner_candidates:
            # calculate distance to candidate
            ddr = euclidean_distance(start.x, start.y, c.x, c.y)

            # recursive
            path, doors = self._find_shortest_paths(c, room, path, doors, dr+ddr)
            path.pop()
        for c in corner_candidates:
            c.unused = True

        for d in door_candidates:
            # calculate distance to candidate
            ddr = euclidean_distance(start.x, start.y, d.inbound_offset.x, d.inbound_offset.y)

            # fix offset
            d.outbound_offset.dx = 0
            d.outbound_offset.dy = 0

            # append to doors list
            doors.append(d)

            # recursive
            path, doors = self._find_shortest_paths(d.outbound_offset, d.outbound_room, path, doors, dr+ddr)

            # pop lists as we're back to current depth
            path.pop()
            doors.pop()

        for d in door_candidates:
            d.unused = True

        return path, doors

    def _populate_grid(self, D, N, M, corner, room):
        """
        Scan matrix and populate with euclidean distance for cells in line of sight of corner.

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
                node = Point(self.x_interp[x], self.y_interp[y])

                # get distance from corner to node if in line of sight
                if not self._has_direct_los(Point(corner.x+corner.dx, corner.y+corner.dy), node):
                    continue

                d = euclidean_distance(corner.x, corner.y, node.x, node.y)

                # update map if d is a valid value
                if d != None:

                    # add distance from sensor to corner
                    d += corner.dmin

                    # update map if less than existing value
                    if D[y, x] == 0 or d < D[y, x]:
                        D[y, x] = d
                        N[y, x] = len(corner.visited_doors)
                        M[y][x] = [door.door_number for door in corner.visited_doors]

        return D, N, M

    def _map_process(self, sensor: Sensor):
        self._reset_pathfinding_variables()

        # Recursively find shortest path from sensor to all corners.
        self._find_shortest_paths(
            start=sensor,
            room=self.rooms[sensor.room_number],
            path=[],
            doors=[],
            dr=0,
        )

        # Initialise necessary grids.
        sensor.D = np.zeros(shape=self.X.shape)
        sensor.N = np.zeros(shape=self.X.shape)
        sensor.M = [[[] for y in range(self.X.shape[1])] for x in range(self.X.shape[0])]

        # Populate map from sensor point of view.
        sensor.D, sensor.N, sensor.M = self._populate_grid(sensor.D, sensor.N, sensor.M, sensor, self.rooms[sensor.room_number])
        # self.plot_debug(start=sensor, grid=[sensor.D])

        # populate grid with distances from each corner
        for ri, room in enumerate(self.rooms):
            # fill from doors
            for di, door in enumerate(self.doors):
                print('Populating distance map: sensor {:>3}, room {:>3},   door {:>3}'.format(sensor.device_id, ri, di))
                if door.outbound_room == room:
                    offset_node = door.outbound_offset
                    if len(offset_node.shortest_path) > 0:
                        sensor.D, sensor.N, sensor.M = self._populate_grid(sensor.D, sensor.N, sensor.M, offset_node, room)

                        # self.plot_debug(start=sensor, grid=[sensor.D], paths=offset_node.shortest_path)

            # fill from corners
            for ci, corner in enumerate(room.corners):
                print('Populating distance map: sensor {:>3}, room {:>3}, corner {:>3}'.format(sensor.device_id, ri, ci))
                if len(corner.shortest_path) > 0:
                    sensor.D, sensor.N, sensor.M = self._populate_grid(sensor.D, sensor.N, sensor.M, corner, room)

                    # self.plot_debug(start=sensor, grid=[sensor.D], paths=corner.shortest_path)

        write_pickle(sensor, f'/tmp/{PICKLE_PREFIX}{sensor.device_id}.pkl')

    def _euclidean_map(self, threaded: bool = True):
        if threaded:
            procs = []
            for room in self.rooms:
                for sensor in room.sensors:
                    p = Process(
                        target=self._map_process,
                        args=(sensor, ),
                    )
                    procs.append(p)
                    procs[-1].start()
                    print(f'Spawned process for {sensor.device_id}')
            
            # Wait for all processes to finish.
            for i, p in enumerate(procs):
                p.join()
                print(f'Completed {i+1}/{len(procs)} processes.')
        else:
            for room in self.rooms:
                for sensor in room.sensors:
                    print(f'Processing {sensor.device_id, }')
                    self._map_process(sensor)

        # Overwrite sensors in rooms with pickled versions from `map_process`.
        for room in self.rooms:
            for i in range(len(room.sensors)):
                room.sensors[i] = read_pickle(
                    path=f'/tmp/{PICKLE_PREFIX}{room.sensors[i].device_id}.pkl',
                )
                
    def initialise_debug_plot(self):
        self.fig, self.ax = plt.subplots()

    def plot_debug(self,
                   start: Point | None = None,
                   goals=None,
                   grid=None,
                   paths=None,
                   show=False,
                   ) -> None:
        # Initialize figure if it is not already open.
        if not hasattr(self, 'ax') or not plt.fignum_exists(self.fig.number):
            self.initialise_debug_plot()

        self.ax.clear()

        for room in self.rooms:
            xx, yy = room.get_outline()
            self.ax.plot(xx, yy, '-k', linewidth=3)

        for door in self.doors:
            self.ax.plot(door.xx, door.yy, '-k', linewidth=14)
            if door.closed:
                self.ax.plot(
                    door.xx,
                    door.yy,
                    '-',
                    color='orangered',
                    linewidth=8,
                )
            else:
                self.ax.plot(
                    door.xx,
                    door.yy,
                    '-',
                    color='limegreen',
                    linewidth=8,
                )

        if goals != None and start != None:
            for g in goals:
                self.ax.plot(
                    [start.x, g.x],
                    [start.y, g.y],
                    '.-r',
                    markersize=10,
                )

        if start is not None:
            self.ax.plot(start.x, start.y, 'ok', markersize=10)

        if paths is not None:
            for i in range(len(paths)-1):
                p1 = paths[i]
                p2 = paths[i+1]
                self.ax.plot([p1.x, p2.x], [p1.y, p2.y], '.-r')

        if grid is not None:
            for g in grid:
                pc = self.ax.contourf(
                    self.X.T,
                    self.Y.T,
                    g.T,
                    max(1, int(g.max()-g.min())),
                )
                pc.set_clim(
                    0,
                    max(
                        self.xlim[1]-self.xlim[0],
                        self.ylim[1]-self.ylim[0],
                    )
                )

        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()
        else:
            plt.waitforbuttonpress()

    def _initialise_heatmap_plot(self):
        self.hfig, self.hax = plt.subplots()
        self.hfig.set_figheight((self.ylim[1] - self.ylim[0]) * 0.5)
        self.hfig.set_figwidth((self.xlim[1] - self.xlim[0]) * 0.5)
        self.hfig.colorbar(
            cm.ScalarMappable(
                norm=Normalize(
                    vmin=T_RANGE[0],
                    vmax=T_RANGE[1]
                ),
                cmap=cm.jet,
            )
        )

    def plot_heatmap(self,
                     timestamp: str = '',
                     blocking: bool = True,
                     show: bool = True,
                     ) -> None:
        # Initialise figure if not already open.
        if not hasattr(self, 'hax') or not plt.fignum_exists(self.hfig.number):
            self._initialise_heatmap_plot()

        self.hax.clear()
        self.hax.set_title(timestamp)

        # Draw all walls that define rooms.
        for room in self.rooms:
            xx, yy = room.get_outline()
            self.hax.plot(xx, yy, '-k', linewidth=3)

        # Draw doors between rooms.
        for door in self.doors:
            self.hax.plot(door.xx, door.yy, '-k', linewidth=14)
            if door.closed:
                self.hax.plot(
                    door.xx,
                    door.yy,
                    '-',
                    color='orangered',
                    linewidth=8,
                )
            else:
                self.hax.plot(
                    door.xx,
                    door.yy,
                    '-',
                    color='limegreen',
                    linewidth=8,
                )

        # Draw sensors as black "X"s on the map.
        for room in self.rooms:
            for sensor in room.sensors:
                self.hax.plot(
                    sensor.x,
                    sensor.y,
                    'xk',
                    markersize=10,
                    markeredgewidth=2.5,
                )

        pc = self.hax.contourf(
            self.X.T,
            self.Y.T,
            self.heatmap.T,
            (T_RANGE[1]-T_RANGE[0])*5,
            cmap=cm.jet,
        )
        pc.set_clim(T_RANGE[0], T_RANGE[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.tight_layout()

        # I noticed that the heatmap doesn't always draw correctly.
        # Found that it helps to give it a second to draw.
        plt.pause(1)

        if blocking:
            if show:
                plt.show()
            else:
                plt.waitforbuttonpress()
        else:
            plt.pause(0.01)

    def stream(self):
        targets = []
        for room in self.rooms:
            for sensor in room.sensors:
                if sensor.device is not None:
                    targets.append(sensor)
        for door in self.doors:
            if door.device is not None:
                targets.append(door)

        if len(np.unique([t.project_id for t in targets])) > 1:
            raise ValueError('Devices must be from same project.')

        self.update_heatmap()
        self.plot_heatmap(
            timestamp=str(datetime.now()),
            blocking=False,
        )

        print('Starting stream ...')
        for event in dt.Stream.event_stream(
            project_id=targets[0].project_id,
            device_ids=[d.device_id for d in targets],
            event_types=[
                dt.events.TEMPERATURE,
                dt.events.HUMIDITY,
                dt.events.OBJECT_PRESENT,
            ],
        ):

            for target in targets:
                if target.device_id == event.device_id:
                    target.update(event)
                    self.update_heatmap()
                    self.plot_heatmap(
                        timestamp=str(datetime.now()),
                        blocking=False,
                    )
                    print(f'Updated {target.device_id}.')
                    break
