# heatmap resolution in points per meter
resolution = 5

# illustrated temperature range in plot
temperature_range = [18, 27]

# list of room parameter dictionaries
rooms = [
    # room 1
    {
        'corners': {
            'x': [0.0, 0.0, 5.5, 5.5],
            'y': [0.0, 4.0, 4.0, 0.0],
        },
    },

    # room 2
    {
        'corners': {
            'x': [5.5, 5.5, 3.5, 3.5, 5.5, 5.5, 25.5, 25.5, 28.0, 28.0, 33.0, 33.0, 23.0, 23.0],
            'y': [0.0, 4.0, 4.0, 8.5, 8.5, 6.0,  6.0,  8.5,  8.5,  6.0,  6.0,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjeho0fbluqg00dltg30',
                'x': 8.0,
                'y': 0.5,
            },
            {
                'id': 'bjeho0fbluqg00dltg30',
                'x': 15.5,
                'y':  0.5,
            },
            {
                'id': 'bddch957rihjnq7k1iu0',
                'x': 12.5,
                'y':  3.5,
            },
            {
                'id': 'bjejnvgpismg008hqrrg',
                'x': 18.0,
                'y':  6.0,
            },
            {
                'id': 'bddch9d7rihjnq7k1mq0',
                'x': 20.5,
                'y':  3.5,
            },
            {
                'id': 'bjei5c67kro000cp0j20',
                'x': 24.0,
                'y':  4.0,
            },
            {
                'id': 'bjei8nvbluqg00dltl20',
                'x': 28.0,
                'y':  7.0,
            },
        ],
    },

    # room 3
    {
        'corners': {
            'x': [11.0, 11.0, 15.5, 15.5, 18.5, 18.5],
            'y': [ 6.0, 11.0, 11.0,  8.5,  8.5,  6.0],
        },
        'sensors': [
            {
                'id': 'bjei8odntbig00e43gr0',
                'x': 11.0,
                'y':  7.5,
            }
        ],
    },

    # room 4
    {
        'corners': {
            'x': [23.0, 23.0, 28.0, 28.0],
            'y': [ 0.0,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjei71tp0jt000aqc78g',
                'x': 25.5,
                'y':  0.0,
            }
        ],
    },

    # room 5
    {
        'corners': {
            'x': [28.0, 28.0, 30.5, 30.5],
            'y': [ 0.0,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjeickgpismg008hqf3g',
                'x': 30.5,
                'y':  0.5,
            },
            {
                'id': 'bjehnpe7gpvg00cjnvv0',
                'x': 30.5,
                'y':  2.0,
            },
            {
                'id': 'bjei8p67kro000cp0k20',
                'x': 30.5,
                'y':  3.5,
            },
        ]
    },

    # room 6
    {
        'corners': {
            'x': [30.5, 30.5, 33.0, 33.0],
            'y': [ 0.0,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjei50vbluqg00dltju0',
                'x': 30.5,
                'y':  2.5,
            }
        ],
    },

    # room 7
    {
        'corners': {
            'x': [33.0, 33.0, 35.5, 35.5],
            'y': [ 0.0,  6.0,  6.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjei8rgpismg008hqdu0',
                'x': 35.5,
                'y':  3.0,
            }
        ],
    },

    # room 8
    {
        'corners': {
            'x': [28.0, 28.0, 31.0, 31.0],
            'y': [ 6.0,  8.5,  8.5,  6.0],
        },
        'sensors': [
            {
                'id': 'bjei75vbluqg00dltkig',
                'x': 30.5,
                'y':  8.5,
            }
        ],
    },
]
