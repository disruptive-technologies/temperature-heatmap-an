# packages
import os
import sys
import pickle
import numpy  as np
import pandas as pd


class Point():
    def __init__(self, x=None, y=None):
        # give to self
        self.x = x
        self.y = y

        # initialise lists
        self.children = []
        self.walls = []

        # flag if used
        self.unused = True

        # variables
        self.shortest_distance = None


class Line():
    def __init__(self, p1, p2, wall=False):
        # give to self
        self.p1 = p1
        self.p2 = p2

        # list of points
        self.pp = [p1, p2]

        # add self object to point
        if wall:
            p1.walls.append(self)
            p2.walls.append(self)

        # some coordinate formatting
        self.xx = [p1.x, p2.x]
        self.yy = [p1.y, p2.y]


class Sensor():
    def __init__(self, p, t=None):
        # give to self
        self.p = p
        self.t = t
        
        # extract x- and y- coordinate
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

