

class Room():
    """
    Class representing a room in layout.
    Can be initialised with complete list of corners and sensors, but 
    is in this application initialised empty and filled later to support .json import.
    Provided name should be unique.

    """

    def __init__(self, corners=[], sensors=[], name=''):
        """
        Room class constructor.

        Parameters
        ----------
        corners : list
            List of one Corner object per corner in room layout.
        sensors : list
            List of one Sensor object per sensor in room layout.

        """

        # initialise variables
        self.corners = corners
        self.sensors = sensors
        self.name    = name


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


class Point():
    """
    Simple base class to represent a point in 2-D space.
    Bare minimum variables. Inherited by other classes.

    """

    def __init__(self, x, y):
        """
        Point class constructor.

        Parameters
        ----------
        x : float
            x-coordinate of point.
        y : float
            y-coordinate of point.

        """

        # give to self
        self.x = x
        self.y = y
        self.dx = 0     # offset in x-direction
        self.dy = 0     # offset in y-direction


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
        self.p1 = p1
        self.p2 = p2

        # some variable formatting for easier access
        self.pp = [p1, p2]
        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


class Corner(Point):
    """
    Class representing a corner in room layout.
    Inherits Point class and adds features.

    """

    def __init__(self, x, y):
        """
        Corner class constructor.

        Parameters
        ----------
        x : float
            x-coordinate of corner position.
        y : float
            y-coordinate of corner position.

        """

        # inherit Point class
        Point.__init__(self, x, y)

        # initialise variables
        self.dmin   = None
        self.unused = True
        self.shortest_path = []


    def set_coordinates(self, x, y):
        """
        Update corner coordinates by re-inheriting Point class.

        Parameters
        ----------
        x : float
            x-coordinate of updated corner position.
        y : float
            y-coordinate of updated corner position.

        """

        # re-inherit Point class with updated values
        Point.__init__(self, x, y)


class Door(Line):
    """
    Class representing a door in room layout.
    Is defined as spanning two Point class objects.
    Inherits Line class.

    """

    def __init__(self):
        """
        Door class constructor.
        Do nothing but set a few variables. Door is initialised later by calling post_initialise().
        This is due to some call-by-reference issues when parsing .json layout.
        Not elegant, but works.

        """

        # initialise variables
        self.closed = False
        self.outbound_room = None

    
    def post_initialise(self, p1, p2, room1, room2, sensor_id, number):
        """
        Door class second-stage constructor.
        Must be called, or object will not function properly.
        Room names (room1/room2) must match name of room objects in layout.

        Parameters
        ----------
        p1 : object
            Point object of where door starts.
        p2 : object
            Point object of where door ends.
        room1 : str
            Name of room on one side of the door.
        room2 : str
            Name of the other room on the other side of the door.
        sensor_id : str
            Sensor id of door proximity sensor.
        number : int
            Door number. Must be unique.

        """

        # inherit Line
        Line.__init__(self, Point(p1[0], p1[1]), Point(p2[0], p2[1]))

        # give to self
        self.p1      = Point(p1[0], p1[1])
        self.p2      = Point(p2[0], p2[1])
        self.room1   = room1
        self.room2   = room2
        self.sensor_id = sensor_id
        self.number  = number

        # variables
        self.unused = True
        self.x = (self.p1.x + self.p2.x) / 2
        self.y = (self.p1.y + self.p2.y) / 2

        # create perpendicular bisector on door line
        self.__perpendicular_bisector(eps=1/1e3)


    def __perpendicular_bisector(self, eps):
        """
        On line beetween points spanning the door, find the perpendicular bisector.

        """

        v = [self.p2.y - self.p1.y, self.p2.x - self.p1.x]
        self.o1 = Corner(self.x+v[0]*(+eps), self.y+v[1]*(+eps))
        self.o2 = Corner(self.x+v[0]*(-eps), self.y+v[1]*(-eps))


    def new_event_data(self, event):
        """
        Update door state from event.
        Served by Director class.

        Parameters
        ----------
        event : dict
            Dictionary of event data containing new door state.

        """

        if event['data']['objectPresent']['state'] == 'PRESENT':
            self.closed = True
        else:
            self.closed = False


class Sensor(Point):
    """
    Class representing a sensor in room layout.
    Inherits Point class.

    """

    def __init__(self, x, y):
        """
        Sensor class constructor.
        Do nothing. Sensor is initialised later by calling post_initialise().
        Same deal as with Door class.

        """

        # inherit point
        Point.__init__(self, x, y)

        # initialise variables
        self.t = None
        self.dmin = None
        self.sensor_id = None


    def post_initialise(self, x, y, sensor_id, room_number):
        """
        Sensor class second-stage constructor.
        Must be called, or object will not function properly.

        Parameters
        ----------
        x : float
            x-coordinate of sensor in room.
        y : float
            y-coordinate of sensor in room.
        sensor_id : str
            Sensor id of sensor. Can be found in DT Studio.
        room_number : id
            Integer representing which room the sensor is in. Not the same as room name.

        """

        # re-inherit point
        Point.__init__(self, x, y)

        # give to self
        self.sensor_id = sensor_id
        self.room_number = room_number


    def new_event_data(self, event):
        """
        Update sensor temperature value from event.
        Served by Director class.

        Parameters
        ----------
        event : dict
            Dictionary of event data containing new temperature value.

        """

        self.t = event['data']['temperature']['value']
