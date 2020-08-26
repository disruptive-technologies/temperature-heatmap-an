# packages
import sys
import numpy             as np
import matplotlib.pyplot as plt

# parameters
res = 5

overall_best = []

class Point():
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.children = []
        self.n = 0
        self.unused = True
        self.walls = []


class Line():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.pp = [p1, p2]

        # add self object to point
        p1.walls.append(self)
        p2.walls.append(self)

        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


points = [
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
    Line(points[0], points[1]),
    Line(points[1], points[2]),
    Line(points[3], points[4]),
    Line(points[5], points[6]),
    Line(points[7], points[8]),
    Line(points[9], points[10]),
    Line(points[11], points[12]),
    Line(points[0], points[3]),
    Line(points[3], points[7]),
    Line(points[7], points[11]),
    Line(points[1], points[6]),
    Line(points[6], points[10]),
    Line(points[2], points[12]),
]


# stat and stop
start = Point( 2,  2)
goal  = Point( 1, 17)

# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r): 
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

# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True

    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False

def point_line_distance(p1, p2, p3):
    return np.linalg.norm(np.cross([p2.x-p1.x, p2.y-p1.y], [p1.x-p3.x, p1.y-p3.y]))/np.linalg.norm([p2.x-p1.x, p2.y-p1.y])


def get_distance(path):
    if len(path) == 0:
        return np.inf
    d = 0
    for i in range(1, len(path)):
        p1 = path[i-1]
        p2 = path[i]
        d += np.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2)
    return d


def check_inward(node, point):
    p0 = point
    x0 = p0.x
    y0 = p0.y

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
    tx = (p1.x + p2.x) / 2
    ty = (p1.y + p2.y) / 2

    plt.cla()
    for wall in point.walls:
        for p in wall.pp:
            plt.plot(p.x, p.y, 'ok')

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



def probe(node, path, best):
    # print('{}, {}'.format(node.x, node.y))

    # check if los to goal
    los = True
    for wall in walls:
        if not node in wall.pp:
            if doIntersect(node, goal, wall.p1, wall.p2):
                los = False
    if los:
        # print('-------------------------- end')
        # path.append(goal)
        # for p in path:
        #     print('[{}, {}]'.format(p.x, p.y))
        # print(get_distance(path))
        # 
        # draw(path=path)

        # path.pop()

        path.append(goal)
        if len(best) == 0 or get_distance(path) < get_distance(best):
            best = []
            for p in path:
                best.append(p)
        path.pop()

        return path, best


    for point in points:
        if node != point and point.unused == True:
            # skip point if zero or more than double intersect
            if len(point.walls) > 2 or len(point.walls) == 0:
                continue

            # check if node is in point walls
            valid = False
            for wall in point.walls:
                if node in wall.pp:
                    valid = True
            if not valid:
                # skip if not in los
                los = True
                for wall in walls:
                    if node not in wall.pp and point not in wall.pp:
                        if doIntersect(node, point, wall.p1, wall.p2):
                            los = False
                if not los:
                    continue

                # skip if inward triangle
                bounded = False
                if len(point.walls) == 2:
                    bounded = check_inward(node, point)
                if bounded:
                    continue

            # recursive
            point.unused = False
            path.append(point)
            path, best = probe(point, path, best)
            # draw(dnode=point, path=path)
            path.pop()
            point.unused = True


    return path, best


def draw(dnode=None, ddirect=None, path=None, cc=0):
    # refresh
    ax.clear()

    # draw line
    for dwall in walls:
        ax.plot(dwall.xx, dwall.yy, color='k', linewidth=3)

    # draw direct
    if ddirect != None:
        ax.plot(ddirect.xx, ddirect.yy, color='r')

    # draw stat and goal
    ax.plot(start.x, start.y, 'or', markersize=25)
    ax.plot(goal.x,  goal.y,  'og', markersize=25)

    # draw node
    if dnode != None:
        ax.plot(dnode.x, dnode.y, 'ok', markersize=15)

    # draw points
    for dpoint in points:
        if dpoint.unused:
            plt.plot(dpoint.x, dpoint.y, 'og', markersize=10)
        else:
            plt.plot(dpoint.x, dpoint.y, 'or', markersize=10)

    if path != None:
        x = [point.x for point in path]
        y = [point.y for point in path]
        ax.plot(x, y, '.-r')

    # set aspect
    if 0:
        plt.gca().set_aspect('equal', adjustable='box')
        plt.waitforbuttonpress()
    else:
        fig.set_figheight(14)
        fig.set_figwidth(18)
        out = '/home/kepler/tmp2/'
        fig.savefig(out + '{:09d}.png'.format(cc), dpi=100, bbox_inches='tight')
        cc += 1
    return cc


if __name__ == '__main__':
    # initialise plot
    fig, ax = plt.subplots()

    cc = 0
    for i in range(1, 14):
        for j in range(1, 18):
            goal.x = i
            goal.y = j

            # keep searching
            path = [start]
            best = []
            path, best = probe(start, path, best)

            cc = draw(path=best, cc=cc)

