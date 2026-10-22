#!/bin/bash
# launch.sh <script-name> <log> : run a harness script detached
nohup setsid /root/work/harness_artts/$1 > /root/work/$2 2>&1 < /dev/null & disown
