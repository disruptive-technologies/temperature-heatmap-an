# packages
import os
import sys
import pickle
import numpy  as np
import pandas as pd


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
    def __init__(self, p1, p2, room1, room2, closed=False):
        # inherit Line
        Line.__init__(self, Point(p1[0], p1[1]), Point(p2[0], p2[1]))

        # give to self
        self.room1  = room1
        self.room2  = room2
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


class Sensor():
    def __init__(self, x, y, name, t0=None):
        # give to self
        self.x    = x
        self.y    = y
        self.name = name
        self.t    = t0

        # make point
        self.p = Point(x, y)


def print_error(text, terminate=True):
    """
    Print an error message and terminate as desired.

    Parameters
    ----------
    terminate : bool
        Terminate execution if True.
    """

    print('ERROR: {}'.format(text))
    if terminate:
        sys.exit()


def convert_event_data_timestamp(ts):
    """
    Convert the default event_data timestamp format to Pandas and unixtime format.

    Parameters
    ----------
    ts : str
        UTC timestamp in custom API event data format.

    Returns
    -------
    timestamp : datetime
        Pandas Timestamp object format.
    unixtime : int
        Integer number of seconds since 1 January 1970.

    """

    timestamp = pd.to_datetime(ts)
    unixtime  = pd.to_datetime(np.array([ts])).astype(int)[0] // 10**9

    return timestamp, unixtime


def json_sort_key(json):
    """
    Return the event update time converted to unixtime.

    Parameters
    ----------
    json : dictionary
        Event data json imported as dictionary.

    Returns
    -------
    unixtime : int
        Event data update time converted to unixtime.

    """

    timestamp = json['data']['temperature']['updateTime']
    _, unixtime = convert_event_data_timestamp(timestamp)
    return unixtime


def loop_progress(i_track, i, n_max, n_steps, name=None, acronym=' '):
    """
    Print loop progress to console.

    Parameters
    ----------
    i_track : int
        Tracks how far the progress has come:
    i : int
        Current index in loop.
    n_max : int
        Maximum value which indicates progress done.
    n_steps : int
        Number of steps to be counted.

    """

    # number of indices in each progress element
    part = n_max / n_steps

    if i_track == 0:
        # print empty bar
        print('    |')
        if name is None:
            print('    └── Progress:')
        else:
            print('    └── {}:'.format(name))
        print('        ├── [ ' + (n_steps-1)*'-' + ' ] ' + acronym)
        i_track = 1
    elif i > i_track + part:
        # update tracker
        i_track = i_track + part

        # print bar
        print('        ├── [ ' + int(i_track/part)*'#' + (n_steps - int(i_track/part) - 1)*'-' + ' ] ' + acronym)

    # return tracker
    return i_track


def eucledian_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def write_pickle(obj, path, cout=True):
    """ write an object to pickle

    arguments:
    obj:    object to be written
    path:   path to which the object is written
    """

    with open(path, 'wb') as output:  # overwrites
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    if cout:
        print('Pickle written to [ ' + path + ' ].')


def read_pickle(path, cout=True):
    """ read a pickle to an object

    arguments:
    path:   path to the pickle
    """

    if os.path.exists(path):
        # load vitals pickle
        try:
            with open(path, 'rb') as input:
                obj = pickle.load(input)
            if cout:
                print('Pickle read from [ ' + path + ' ].')
            return obj
        except ValueError:
            print('\nCould not load pickle. Try other python 3.x')
        except UnicodeDecodeError:
            print('\nCould not load pickle. Try other python 2.x')
        except FileNotFoundError:
            print('\nCould not find any vitals pickle file.')
    else:
        print('\nCould not find pickle at [ ' + path + ' ]')
    print('Terminating...\n')
    sys.exit()

