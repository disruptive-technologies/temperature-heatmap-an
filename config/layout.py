# project
from heatmap.helpers import Point
from heatmap.helpers import Line
from heatmap.helpers import Sensor

# corners
corners = [
    Point( 0.0,  0.0),  #  0
    Point( 0.0,  4.0),  #  1
    Point( 5.5,  0.0),  #  2
    Point( 5.5,  4.0),  #  3
    Point( 4.5,  4.0),  #  4
    Point( 3.5,  4.0),  #  5
    Point( 3.5,  8.5),  #  6
    Point( 5.5,  8.5),  #  7
    Point( 5.5,  6.0),  #  8
    Point(11.0,  6.0),  #  9
    Point(11.0, 11.0),  # 10
    Point(15.5, 11.0),  # 11
    Point(15.5,  8.5),  # 12
    Point(18.5,  8.5),  # 13
    Point(18.5,  6.0),  # 14
    Point(16.0,  6.0),  # 15
    Point(17.0,  6.0),  # 16
    Point(25.5,  6.0),  # 17
    Point(25.5,  8.5),  # 18
    Point(28.0,  8.5),  # 19
    Point(28.0,  6.0),  # 20
    Point(31.0,  6.0),  # 21
    Point(31.0,  8.5),  # 22
    Point(33.0,  6.0),  # 23
    Point(35.5,  6.0),  # 24
    Point(35.5,  0.0),  # 25
    Point(33.0,  0.0),  # 26
    Point(33.0,  4.0),  # 27
    Point(30.5,  4.0),  # 28
    Point(28.0,  4.0),  # 29
    Point(23.0,  4.0),  # 30
    Point(23.0,  0.0),  # 31
]

# walls
walls = [
    Line(corners[0],  corners[1] , wall=True), #  0
    Line(corners[0],  corners[2] , wall=True), #  1
    Line(corners[2],  corners[3] , wall=True), #  2
    Line(corners[4],  corners[5] , wall=True), #  3
    Line(corners[1],  corners[5] , wall=True), #  4
    Line(corners[5],  corners[6] , wall=True), #  5
    Line(corners[6],  corners[7] , wall=True), #  6
    Line(corners[7],  corners[8] , wall=True), #  7
    Line(corners[8],  corners[9] , wall=True), #  8
    Line(corners[9],  corners[10], wall=True), #  9
    Line(corners[10], corners[11], wall=True), # 10
    Line(corners[11], corners[12], wall=True), # 11
    Line(corners[12], corners[13], wall=True), # 12
    Line(corners[13], corners[14], wall=True), # 13
    Line(corners[9],  corners[15], wall=True), # 14
    Line(corners[16], corners[14], wall=True), # 15
    Line(corners[14], corners[17], wall=True), # 16
    Line(corners[17], corners[18], wall=True), # 17
    Line(corners[18], corners[19], wall=True), # 18
    Line(corners[19], corners[20], wall=True), # 19
    Line(corners[20], corners[21], wall=True), # 20
    Line(corners[21], corners[22], wall=True), # 21
    Line(corners[19], corners[22], wall=True), # 22
    Line(corners[21], corners[23], wall=True), # 23
    Line(corners[23], corners[27], wall=True), # 24
    Line(corners[23], corners[24], wall=True), # 25
    Line(corners[24], corners[25], wall=True), # 26
    Line(corners[25], corners[26], wall=True), # 27
    Line(corners[26], corners[27], wall=True), # 28
    Line(corners[27], corners[28], wall=True), # 29
    Line(corners[28], corners[29], wall=True), # 30
    Line(corners[29], corners[30], wall=True), # 31
    Line(corners[30], corners[31], wall=True), # 32
    Line(corners[31], corners[2] , wall=True), # 33
]

# sensors
sensors = [
    # Sensor(Point( 8.0,  0.5)), # 1
    # Sensor(Point(12.5,  3.5)), # 2
    Sensor(Point(11.1,  7.5)), # 2
]
