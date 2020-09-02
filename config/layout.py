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
            Sensor(x=8.0, y=0.5, name='bjeho0fbluqg00dltg30')
        ],
    ),
]

doors = [
    Door(p1=[4.5, 4.0], p2=[5.5, 4.0], room1=rooms[0], room2=rooms[1]),
]
