# project
from heatmap.director import Director

# Fill in from the Service Account and Project:
USERNAME   = "bsvsebr24spg00b250ag"       # this is the key
PASSWORD   = "09e436a4ab3e44779b36ded5f5ff1386"     # this is the secret
PROJECT_ID = "br793014jplfqcpoj45g"                # this is the project id

# url base and endpoint
API_URL_BASE  = "https://api.disruptive-technologies.com/v2"


if __name__ == '__main__':

    # initialise Director instance
    d = Director(USERNAME, PASSWORD, PROJECT_ID, API_URL_BASE)

    # iterate historic events
    d.run_history()

    # stream realtime events
    # d.run_stream(n_reconnects=5)

