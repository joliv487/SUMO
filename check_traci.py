import os
import sys

SUMO_HOME = os.environ.get("SUMO_HOME")

if not SUMO_HOME:
    raise SystemExit("SUMO_HOME not set")

sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

print("TraCI connection works")