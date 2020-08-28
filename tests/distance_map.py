# packages
import sys
import numpy             as np
import matplotlib.pyplot as plt


class Point():
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.children = []
        self.unused = True
        self.walls = []

class Line():
    def __init__(self, p1, p2, wall=False):
        self.p1 = p1
        self.p2 = p2
        self.pp = [p1, p2]

        # add self object to point
        if wall:
            p1.walls.append(self)
            p2.walls.append(self)

        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


class Sensor():
    def __init__(self, p):
        self.p = p
        self.x = p.x
        self.y = p.y
        

if 0:
    # parameters
    corners = [
        Point(0,0),
        Point(10,0),
        Point(14,0),
        Point(0,10),
        Point(6,10),
        Point(8,10),
        Point(10,10),
        Point(0,15),
        Point(2,15),
        Point(4,15),
        Point(10,15),
        Point(0,18),
        Point(14,18),
    ]
    
    walls = [
        Line(corners[0],  corners[1] , wall=True),
        Line(corners[1],  corners[2] , wall=True),
        Line(corners[3],  corners[4] , wall=True),
        Line(corners[5],  corners[6] , wall=True),
        Line(corners[7],  corners[8] , wall=True),
        Line(corners[9],  corners[10], wall=True),
        Line(corners[11], corners[12], wall=True),
        Line(corners[0],  corners[3] , wall=True),
        Line(corners[3],  corners[7] , wall=True),
        Line(corners[7],  corners[11], wall=True),
        Line(corners[1],  corners[6] , wall=True),
        Line(corners[6],  corners[10], wall=True),
        Line(corners[2],  corners[12], wall=True),
    ]
    
    sensors = [
        Sensor(Point(12, 16)),
    ]

else:
    # project
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.layout import corners
    from config.layout import walls
    from config.layout import sensors


class Director():
    def __init__(self, corners, walls, sensors, resolution):
        # give to self
        self.corners    = corners
        self.walls      = walls
        self.sensors    = sensors
        self.resolution = resolution
        self.cc = 0
        self.dd = 0

        # initialise plot
        self.initialise_plot()

        # get limits for x- and y- axes
        self.__set_bounding_box()

        # generate meshgrid
        self.__generate_meshgrid()

        # pre-calculate sensor distances in grid
        self.__precalculate_eucledian()


    def initialise_plot(self):
        self.fig, self.ax = plt.subplots()


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


    def __generate_meshgrid(self):
        # generate interpolation axes
        self.x_interp = np.linspace(self.xlim[0], self.xlim[1], int(self.resolution*(self.xlim[1]-self.xlim[0])+0.5))
        self.y_interp = np.linspace(self.ylim[0], self.ylim[1], int(self.resolution*(self.ylim[1]-self.ylim[0])+0.5))

        # convert to compatible grid
        self.X, self.Y = np.meshgrid(self.x_interp, self.y_interp)


    def eucledian_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)


    def __precalculate_eucledian(self):
        if 1:
            # iterate sensors
            for i, s in enumerate(self.sensors):
                # reset corner distances
                for corner in corners:
                    corner.shortest_distance = np.inf

                # initialise empty distance grid
                s.D = np.zeros(shape=self.X.shape)
                path = []

                self.dd = 0
                s.D, _ = self.__fill_grid(s.D, s.p, s.p, path, dr=0)
                self.plot(start=s, grid=[s.D])

        else:
            for x in range(self.xlim[0], self.xlim[1]):
                for y in range(self.ylim[0], self.ylim[1]):
                    if (x+1)%2 == 0:
                        yy = self.ylim[1] - y - 0.5
                    else:
                        yy = y + 0.5
                    xx = x + 0.5

                    s = Sensor(Point(xx, yy))

                    # initialise empty distance grid
                    s.D = np.zeros(shape=self.X.shape)
                    path = []

                    s.D, _ = self.__fill_grid(s.D, s.p, path, dr=0)
                    self.plot(start=s, grid=[s.D])


    def __fill_grid(self, D, initial, origin, path, dr):
        print(self.dd)
        path.append([origin.x, origin.y])
        # iterate grid
        # iterate x-axis
        for x, gx in enumerate(self.x_interp):
            # iterate y-axis
            for y, gy in enumerate(self.y_interp):
                # iterate sensors
                # get distance from sensor to grid node
                node = Point(self.x_interp[x], self.y_interp[y])

                d = self.__pathfind(origin, node)
                if d != None:
                    d = d + dr
                    if D[y, x] == 0 or d < D[y, x]:
                        D[y, x] = d
            # self.plot(start=origin, goal=node, grid=[D])

        # find first corner candidates
        candidates = []
        # nudge in convex directions
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            # create offset point
            m = 10
            offset = Point(origin.x+dx/m, origin.y+dy/m)
            # validate nudge is convex
            if not self.__is_concave(offset, origin):
                for corner in corners:
                    # validate corner
                    if corner.unused:
                        if self.__validate_corner(offset, corner):
                            candidates.append(corner)
                            corner.unused = False
        
        if 1:
            # plot
            self.plot(start=origin, goal=node, grid=[D], candidates=candidates, path=path)

        # recursively iterate candidates
        for c in candidates:
            # calculate distance to candidate
            ddr = self.eucledian_distance(origin.x, origin.y, c.x, c.y)

            # if dr + ddr < c.shortest_distance:
            if 1:
                c.shortest_distance = dr + ddr
                # recursive
                # c.unused = False
                self.dd += 1
                D, path = self.__fill_grid(D, initial, c, path, dr+ddr)
                path.pop()
        for c in candidates:
            c.unused = True
        
        return D, path


    def __validate_corner(self, origin, corner):
        # skip if more than 2 branches
        if len(corner.walls) > 2 or len(corner.walls) == 0:
            return False
        
        # skip if concave
        if len(corner.walls) == 2 and self.__is_concave(origin, corner):
            print('[{}, {}]'.format(corner.x, corner.y))
            return False

        # skip if corner is relative to origin
        if len(origin.walls) == 2 and self.__is_concave(corner, origin):
            return False

        # skip is no los
        for wall in self.walls:
            if corner not in wall.pp and origin not in wall.pp:
                if self.line_intersects(origin, corner, wall.p1, wall.p2):
                    return False
        
        # passed
        return True


    def __pathfind(self, start, goal):
        # draw a straight line
        straight = Line(start, goal)

        # check if los
        los = True
        for wall in self.walls:
            if start not in wall.pp:
                if self.line_intersects(start, goal, wall.p1, wall.p2):
                    los = False
                # concave check
                elif len(start.walls) == 2 and self.__is_concave(goal, start):
                    los = False
        if los:
            return self.eucledian_distance(start.x, start.y, goal.x, goal.y)
        return None


    def plot(self, start=None, goal=None, lines=None, grid=None, candidates=None, path=None):
        # clear
        self.ax.clear()

        # draw walls
        for wall in self.walls:
            self.ax.plot(wall.xx, wall.yy, '-k', linewidth=10)

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
                pc.set_clim(0, 30)

        # plot path
        if path != None:
            for i in range(1, len(path)):
                xx = [path[i-1][0], path[i][0]]
                yy = [path[i-1][1], path[i][1]]
                self.ax.plot(xx, yy, '.-r')

        if 0:
            self.fig.set_figheight(self.ylim[1] - self.ylim[0])
            self.fig.set_figwidth(self.xlim[1] - self.xlim[0])
            out = '/home/kepler/tmp3/'
            self.fig.savefig(out + '{:09d}.png'.format(self.cc), dpi=100, bbox_inches='tight')
            self.cc += 1
        else:
            plt.gca().set_aspect('equal', adjustable='box')
            plt.waitforbuttonpress()
            

    def __on_segment(self, p, q, r): 
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
               (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False
  

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


    def line_intersects(self, p1,q1,p2,q2): 
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


    def point_line_distance(self, p1, p2, p3):
        return np.linalg.norm(np.cross([p2.x-p1.x, p2.y-p1.y], [p1.x-p3.x, p1.y-p3.y]))/np.linalg.norm([p2.x-p1.x, p2.y-p1.y])


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


if __name__ == '__main__':
    # initialise director
    d = Director(corners, walls, sensors, resolution=3)


