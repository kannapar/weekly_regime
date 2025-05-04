#Configuration File
from datetime import datetime as dt
import numpy as np
import pandas as pd

### DATA VARIABLES

# Data start date (YYYY,MM,DD)
StartDate = dt(2015,4,28)
EndDate = dt(2025,5,4)
# Data Frequency -> Daily-252, Weekly- 52, Monthly- 12, Annual- 1
StepsPerYear = 52
# Return Prices/Returns
ReturnRets = True

## DEFINE REGIME VARIABLES
# Number of Regimes
NumRegime = 2
Lambda = 0.08
# regime numerical Values (Normal/Crash/Transition)
NormalRegimeVal = 1
CrashRegimeVal = -1
TransitionRegimeVal = 0
#to forecast regime path
CurrentRegime = NormalRegimeVal