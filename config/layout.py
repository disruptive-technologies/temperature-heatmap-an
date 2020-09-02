# project
from heatmap.helpers import Room
from heatmap.helpers import Corner
from heatmap.helpers import Door
from heatmap.helpers import Sensor


rooms = [
    # room 0
    Room(
        corners = [
            Corner(x=0.0, y=0.0),
            Corner(x=0.0, y=4.0),
            Corner(x=5.5, y=4.0),
            Corner(x=5.5, y=0.0),
        ],
        sensors = [

        ],
    ),

    # room 1
    Room(
        corners = [
            Corner(x= 5.5, y=0.0),
            Corner(x= 5.5, y=4.0),
            Corner(x= 3.5, y=4.0),
            Corner(x= 3.5, y=8.5),
            Corner(x= 5.5, y=8.5),
            Corner(x= 5.5, y=6.0),
            Corner(x=25.5, y=6.0),
            Corner(x=25.5, y=8.5),
            Corner(x=28.0, y=8.5),
            Corner(x=28.0, y=6.0),
            Corner(x=33.0, y=6.0),
            Corner(x=33.0, y=4.0),
            Corner(x=23.0, y=4.0),
            Corner(x=23.0, y=0.0),
        ],
        sensors = [
            Sensor(x=8.0, y=0.5, name='bjeho0fbluqg00dltg30', t0=21)
        ],
    ),

    # room 2
    Room(
        corners = [
            Corner(x=11.0, y= 6.0),
            Corner(x=11.0, y=11.0),
            Corner(x=15.5, y=11.0),
            Corner(x=15.5, y= 8.5),
            Corner(x=18.5, y= 8.5),
            Corner(x=18.5, y= 6.0),
        ],
        sensors = [
            Sensor(x=11.1, y=7.5, name='bjei8odntbig00e43gr0', t0=22)
        ],
    ),

    # room 3
    Room(
        corners = [
            Corner(x=23.0, y= 0.0),
            Corner(x=23.0, y= 4.0),
            Corner(x=28.0, y= 4.0),
            Corner(x=28.0, y= 0.0),
        ],
        sensors = [
            Sensor(x=25.5, y=0.5, name='bjei71tp0jt000aqc78g', t0=23)
        ],
    ),

    # room 4
    Room(
        corners = [
            Corner(x=28.0, y= 0.0),
            Corner(x=28.0, y= 4.0),
            Corner(x=30.5, y= 4.0),
            Corner(x=30.5, y= 0.0),
        ],
        sensors = [

        ],
    ),

    # room 5
    Room(
        corners = [
            Corner(x=30.5, y= 0.0),
            Corner(x=30.5, y= 4.0),
            Corner(x=33.0, y= 4.0),
            Corner(x=33.0, y= 0.0),
        ],
        sensors = [
            Sensor(x=30.6, y= 2.0, name='bjei50vbluqg00dltju0', t0=24)
        ],
    ),

    # room 6
    Room(
        corners = [
            Corner(x=33.0, y= 0.0),
            Corner(x=33.0, y= 6.0),
            Corner(x=35.5, y= 6.0),
            Corner(x=35.5, y= 0.0),
        ],
        sensors = [
            Sensor(x=35.4, y= 3.0, name='bjei8rgpismg008hqdu0', t0=25)
        ],
    ),

    # room 7
    Room(
        corners = [
            Corner(x=28.0, y= 6.0),
            Corner(x=28.0, y= 8.5),
            Corner(x=31.0, y= 8.5),
            Corner(x=31.0, y= 6.0),
        ],
        sensors = [
            Sensor(x=30.5, y= 8.4, name='bjei75vbluqg00dltkig', t0=26)
        ],
    ),
]

doors = [
    Door(p1=[ 4.5, 4.0], p2=[ 5.5, 4.0], room1=rooms[0], room2=rooms[1], closed=False),
    Door(p1=[16.0, 6.0], p2=[17.0, 6.0], room1=rooms[1], room2=rooms[2], closed=False),
    Door(p1=[27.0, 4.0], p2=[28.0, 4.0], room1=rooms[1], room2=rooms[3], closed=False),
    Door(p1=[29.5, 4.0], p2=[30.5, 4.0], room1=rooms[1], room2=rooms[4], closed=False),
    Door(p1=[30.5, 4.0], p2=[31.5, 4.0], room1=rooms[1], room2=rooms[5], closed=False),
    Door(p1=[33.0, 6.0], p2=[33.0, 5.0], room1=rooms[1], room2=rooms[6], closed=False),
    Door(p1=[30.0, 6.0], p2=[31.0, 6.0], room1=rooms[1], room2=rooms[7], closed=True),
]
