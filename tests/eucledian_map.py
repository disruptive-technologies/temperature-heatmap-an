# packages
import os
import sys
import argparse
import numpy             as np
import matplotlib.pyplot as plt

# mutlithreading
import multiprocessing as mpr

# project pathing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# import layout
from config.layout import corners
from config.layout import walls
from config.layout import sensors

# project imports
import heatmap.helpers as hlp

# pathing
pickle_identifier = 'heatmap_'
cache_dir = '/tmp/'


class Director():
    def __init__(self, corners, walls, sensors, resolution):
        # give to self
        self.corners    = corners
        self.walls      = walls
        self.sensors    = sensors
        self.resolution = resolution
        self.n_sensors  = len(sensors)
        self.cc = 0
        self.dd = 0

        # parse system arguments
        self.__parse_sysargs()

        # initialise plot
        self.initialise_plot()

        # get limits for x- and y- axes
        self.__set_bounding_box()

        # generate meshgrid
        self.__generate_meshgrid()

        # pre-calculate sensor distances in grid
        if self.args['debug']:
            self.__eucledian_map_debug()
        else:
            self.__eucledian_map_threaded()

        # Plot results
        for sensor in self.sensors:
            self.plot(start=sensor.p, grid=[sensor.D])


    def __parse_sysargs(self):
        """
        Parse for command line arguments.

        """

        # create parser object
        parser = argparse.ArgumentParser(description='Some explanatory text here.')

        # general arguments
        # parser.add_argument('--timestep',  metavar='', help='Heatmap update period.', required=False, default=3600, type=int)

        # boolean flags
        parser.add_argument('--debug', action='store_true', help='Turns off things like multithreading for easier access.')
        parser.add_argument('--read',  action='store_true', help='Read cached distance maps.')

        # convert to dictionary
        self.args = vars(parser.parse_args())


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
            for corner in corners:
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
            if 1:
                self.plot(start=sensor.p, grid=[sensor.D])

            # populate grid with distances from each corner
            for ci, corner in enumerate(corners):
                if len(corner.shortest_path) > 0:
                    print('sensor {}, corner {}'.format(i, ci))
                    sensor.D = self.__populate_grid(sensor.D, corner)

                    # plot population process
                    if 1:
                        self.plot(goal=corner, grid=[sensor.D], paths=[corner.shortest_path])

            # plot population result
            self.plot(start=sensor.p, grid=[sensor.D])


    def __eucledian_map_threaded(self):
        def map_process(sensor, i):
            # reset corner distances
            for corner in corners:
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
            for ci, corner in enumerate(corners):
                if len(corner.shortest_path) > 0:
                    print('Populating map for sensor {} at corner {}.'.format(i, ci))
                    sensor.D = self.__populate_grid(sensor.D, corner)

            # write sensor object to pickle
            hlp.write_pickle(sensor, os.path.join(cache_dir, pickle_identifier + '{}.pkl'.format(i)), cout=True)

        # just skip everything and read from cache if so desired
        if self.args['read']:
            self.sensors = self.__get_cached_sensors()
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
        self.sensors = self.__get_cached_sensors()


    def __get_cached_sensors(self):
        # initialise list
        sensors = []

        # get files in cache
        cache_files = os.listdir(cache_dir)

        # iterate sensors
        for i in range(self.n_sensors):
            # keep track of if we found the pickle
            found = False

            # iterate files in cache
            for f in cache_files:
                # look for correct pickle
                if pickle_identifier + '{}.pkl'.format(i) in f and not found:
                    # read pickle
                    pickle_sensor = hlp.read_pickle(os.path.join(cache_dir, pickle_identifier + '{}.pkl'.format(i)), cout=True)

                    # exchange
                    sensors.append(pickle_sensor)

                    # found it
                    found = True

            # shouldn't happen, but just in case
            if not found:
                hlp.print_error('Pickle #{} has gone missing.'.format(i), terminate=True)

        return sensors


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
                for corner in corners:
                    # skip used
                    if corner.unused:
                        # validate corner
                        ddr = self.eucledian_distance(active.x, active.y, corner.x, corner.y)
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
            ddr = self.eucledian_distance(active.x, active.y, c.x, c.y)

            # recursive
            self.dd += 1
            path = self.__find_shortest_paths(c, path, dr+ddr)
            path.pop()
        for c in candidates:
            c.unused = True

        return path


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
                    d = self.maxdim - (d + corner.shortest_distance)

                    # update map if less than existing value
                    if d > D[y, x]:
                        D[y, x] = d

        return D


    def __pathfind(self, start, goal):
        # draw a straight line
        straight = hlp.Line(start, goal)

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
                if self.line_intersects(origin, corner, wall.p1, wall.p2):
                    return False
        
        # passed
        return True


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


    def eucledian_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)


    def initialise_plot(self):
        self.fig, self.ax = plt.subplots()


    def plot(self, start=None, goal=None, lines=None, grid=None, candidates=None, paths=None):
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
            out = '/home/kepler/tmp3/'
            self.fig.savefig(out + '{:09d}.png'.format(self.cc), dpi=100, bbox_inches='tight')
            self.cc += 1
        else:
            plt.gca().set_aspect('equal', adjustable='box')
            plt.waitforbuttonpress()


if __name__ == '__main__':
    # initialise director
    d = Director(corners, walls, sensors, resolution=3)

