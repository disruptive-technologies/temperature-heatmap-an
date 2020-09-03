

class Room():
    def __init__(self, corners, sensors):
        # give to self
        self.corners = corners
        self.sensors = sensors


    def get_outline(self):
        xax = [c.x for c in self.corners] + [self.corners[0].x]
        yax = [c.y for c in self.corners] + [self.corners[0].y]
        return xax, yax


class Point():
    def __init__(self, x, y):
        # give to self
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.dmin = None


class Line():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.pp = [p1, p2]
        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


class Corner(Point):
    def __init__(self, x, y):
        # inherit point
        Point.__init__(self, x, y)

        # variables
        self.unused = True
        self.shortest_path = []


class Door():
    def __init__(self, p1, p2, room1, room2, name=None, closed=False):
        # inherit Line
        Line.__init__(self, Point(p1[0], p1[1]), Point(p2[0], p2[1]))

        # give to self
        self.room1  = room1
        self.room2  = room2
        self.name   = name
        self.closed = closed

        # variables
        self.unused = True
        self.x = (self.p1.x + self.p2.x) / 2
        self.y = (self.p1.y + self.p2.y) / 2

        # create perpendicular bisectors
        v = [self.p2.y - self.p1.y, self.p2.x - self.p1.x]
        eps = 1/10
        self.o1 = Corner(self.x+v[0]*(+eps), self.y+v[1]*(+eps))
        self.o2 = Corner(self.x+v[0]*(-eps), self.y+v[1]*(-eps))


    def new_event_data(self, event):
        if event['data']['objectPresent']['state'] == 'PRESENT':
            self.closed = True
        else:
            self.closed = False


class Sensor():
    def __init__(self, x, y, name, t0=None):
        # give to self
        self.x    = x
        self.y    = y
        self.name = name
        self.t    = t0

        # make point
        self.p = Point(x, y)


    def new_event_data(self, event):
        self.t = event['data']['temperature']['value']
