# Streaming Temperature Heatmap

## What am I?
This repository contains the example code talked about in [this application note (LINK PENDING)](https://www.disruptive-technologies.com/), presenting a method of using the Disruptive Technologies (DT) Wireless Temperature Sensors for generating a temperature heatmap which are continously updated for a data stream. Written in Python 3, it uses the DT Developer API to communicate with a DT Studio project and its sensors. 

## Before Running Any code
A DT Studio project containing temperature sensors should be made. All temperature sensors in the project will attemped forecasted upon.

## Environment Setup
Dependencies can be installed using pip.
```
pip3 install -r requirements.txt
```

Edit *sensor_stream.py* to provide the following authentication details of your project. Information about setting up your project for API authentication can be found in this [streaming API guide](https://support.disruptive-technologies.com/hc/en-us/articles/360012377939-Using-the-stream-API).
```python
USERNAME   = "SERVICE_ACCOUNT_KEY"       # this is the key
PASSWORD   = "SERVICE_ACCOUT_SECRET"     # this is the secret
PROJECT_ID = "PROJECT_ID"                # this is the project id
```

## Usage
Running *python3 sensor_stream.py* without any arguments will generate a sample room- and sensor layout which will be displayed. A custom layout can be used by providing a .json file with the --layout argument. More details about generating custom layouts can be found in the application note.
```
usage: sensor_stream.py [-h] [--layout] [--starttime] [--endtime] [--timestep]
                        [--no-plot] [--debug] [--read]

Heatmap generation on Stream and Event History.

optional arguments:
  -h, --help    show this help message and exit
  --layout      Json file with room layout.
  --starttime   Event history UTC starttime [YYYY-MM-DDTHH:MM:SSZ].
  --endtime     Event history UTC endtime [YYYY-MM-DDTHH:MM:SSZ].
  --timestep    Heatmap update period.
  --no-plot     Suppress plots in stream.
  --debug       Disables multithreading for debug visualization.
  --read        Import cached distance maps.
```

Note: When using the *--starttime* argument for a date far back in time, if many sensors exist in the project, the paging process might take several minutes.

