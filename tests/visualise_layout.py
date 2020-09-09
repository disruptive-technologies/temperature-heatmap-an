# packages
import os
import sys

# project pathing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# project imports
from heatmap.director import Director


if __name__ == '__main__':
    # initialise director
    d = Director(resolution=5)

    # plot bare minimums
    d.plot_debug()

