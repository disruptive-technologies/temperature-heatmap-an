# packages
import sys
import numpy  as np
import pandas as pd


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


def eucledian_distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
