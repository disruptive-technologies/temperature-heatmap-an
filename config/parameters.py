# heatmap resolution in points per meter
resolution = 5

# illustrated temperature range in plot
temperature_range = [18, 27]

# list of room parameter dictionaries
rooms = [
    # room 0
    {
        'corners': {
            'x': [0.0, 0.0, 5.0, 5.0],
            'y': [0.0, 3.6, 3.6, 0.0],
        },
    },

    # room 1
    {
        'corners': {
            'x': [5.0, 5.0, 3.3, 3.3, 5.0, 5.0, 23.8, 23.8, 26.0, 26.0, 31.2, 31.2, 21.3, 21.3],
            'y': [0.0, 3.6, 3.6, 7.9, 7.9, 5.6,  5.6,  7.9,  7.9,  5.6,  5.6,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjei55fbluqg00dltjv0',
                'x': 14.3,
                'y':  0.0,
            },
            {
                'id': 'bjei5c67kro000cp0j20',
                'x': 22.0,
                'y':  4.0,
            },
            {
                'id': 'bjeho0fbluqg00dltg30',
                'x': 7.3,
                'y': 0.0,
            },
            {
                'id': 'bjejnvgpismg008hqrrg',
                'x': 17.0,
                'y':  5.6,
            },
            {
                'id': 'bjei8nvbluqg00dltl20',
                'x': 26.0,
                'y':  6.7,
            },
            {
                'id': 'bddch9d7rihjnq7k1mq0',
                'x': 19.8,
                'y':  3.0,
            },
            {
                'id': 'bddch957rihjnq7k1iu0',
                'x': 11.7,
                'y': 3.4,
            },
        ],
    },

    # room 2
    {
        'corners': {
            'x': [21.3, 21.3, 26.0, 26.0],
            'y': [ 0.0,  4.0,  4.0,  0.0],
        },
        'sensors': [
            {
                'id': 'bjei71tp0jt000aqc78g',
                'x': 24,
                'y':  0,
            }
        ]
    },
]
