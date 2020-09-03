# project
from heatmap.miniclasses import Room
from heatmap.miniclasses import Corner
from heatmap.miniclasses import Door
from heatmap.miniclasses import Sensor


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
            Sensor(x=3.0, y=0.1, name='bjeicju7kro000cp0l8g', t0=None),
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
            Sensor(x= 3.6, y=7.0, name='bfuj6v8lq8r000ela7cg', t0=None),
            Sensor(x= 8.0, y=0.5, name='bjeho0fbluqg00dltg30', t0=None),
            Sensor(x=14.5, y=3.5, name='bhnc1ghqitfg008o31ag', t0=None),
            Sensor(x=14.5, y=0.1, name='bfuj719o5b7g0093bbfg', t0=None),
            # Sensor(x=18.0, y=5.9, name='bjejnvgpismg008hqrrg', t0=None),
            Sensor(x=24.0, y=4.1, name='bjei5c67kro000cp0j20', t0=None),
            Sensor(x=27.9, y=7.0, name='bjei8nvbluqg00dltl20', t0=None),
            Sensor(x=21.0, y=4.0, name='bfuj70olq8r000ela7dg', t0=None),
            Sensor(x=22.9, y=0.5, name='bfuj75ho5b7g0093bbk0', t0=None),
            Sensor(x=17.0, y=4.0, name='bfui2sglq8r000el9htg', t0=None),
            Sensor(x=32.7, y=4.1, name='bfuj701o5b7g0093bbeg', t0=None),
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
            Sensor(x=11.1, y=7.5, name='bjei8odntbig00e43gr0', t0=None),
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
            Sensor(x=27.9, y=0.1, name='bjei71tp0jt000aqc78g', t0=None),
            Sensor(x=23.1, y=3.5, name='bfui35ho5b7g0093am6g', t0=None),
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
            Sensor(x=30.4, y= 3.0, name='bjei8p67kro000cp0k20', t0=None),
            Sensor(x=30.4, y= 2.0, name='bjehnpe7gpvg00cjnvv0', t0=None),
            Sensor(x=30.4, y= 1.0, name='bjeickgpismg008hqf3g', t0=None),
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
            Sensor(x=30.6, y= 2.0, name='bjei50vbluqg00dltju0', t0=None),
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
            Sensor(x=35.4, y= 3.0, name='bjei8rgpismg008hqdu0', t0=None),
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
            Sensor(x=30.5, y= 8.4, name='bjei75vbluqg00dltkig', t0=None),
        ],
    ),

    # room 8
    Room(
        corners = [
            Corner(x=5.5, y= 6.0),
            Corner(x=5.5, y= 8.5),
            Corner(x=8.0, y= 8.5),
            Corner(x=8.0, y= 6.0),
        ],
        sensors = [
            Sensor(x=6.75, y=8.4, name='bjei4vm7gpvg00cjo3rg', t0=None),
        ],
    ),

    # room 9
    Room(
        corners = [
            Corner(x=31.0, y= 6.0),
            Corner(x=31.0, y=11.0),
            Corner(x=34.5, y=11.0),
            Corner(x=34.5, y= 6.0),
        ],
        sensors = [
            Sensor(x=31.1, y=8.5, name='bjei50dp0jt000aqc6kg', t0=None),
            Sensor(x=34.4, y=6.1, name='bfui31ho5b7g0093am2g', t0=None),
        ],
    ),
]

doors = [
    Door(p1=[ 4.5, 4.0], p2=[ 5.5, 4.0], room1=rooms[0], room2=rooms[1], name='bh3d3dl7rihlu0c3dm10', closed=False),
    Door(p1=[16.0, 6.0], p2=[17.0, 6.0], room1=rooms[1], room2=rooms[2], name='bh3d3ed7rihlu0c3du90', closed=False),
    Door(p1=[27.0, 4.0], p2=[28.0, 4.0], room1=rooms[1], room2=rooms[3], name='bh3d3dl7rihlu0c3djs0', closed=False),
    Door(p1=[29.5, 4.0], p2=[30.5, 4.0], room1=rooms[1], room2=rooms[4], name='bh3d3dl7rihlu0c3djvg', closed=False),
    Door(p1=[30.5, 4.0], p2=[31.5, 4.0], room1=rooms[1], room2=rooms[5], name='bh3d3dt7rihlu0c3dolg', closed=False),
    Door(p1=[33.0, 6.0], p2=[33.0, 5.0], room1=rooms[1], room2=rooms[6], name='bh3d3f57rihlu0c3e5mg', closed=False),
    Door(p1=[30.0, 6.0], p2=[31.0, 6.0], room1=rooms[1], room2=rooms[7], name='bh3d3cl7rihlu0c3d820', closed=False),
    Door(p1=[ 7.0, 6.0], p2=[ 8.0, 6.0], room1=rooms[1], room2=rooms[8], name='bh3d3dt7rihlu0c3dp1g', closed=False),
    Door(p1=[31.5, 6.0], p2=[32.5, 6.0], room1=rooms[1], room2=rooms[9], name='bh3d3dd7rihlu0c3dif0', closed=False),
]
