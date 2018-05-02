# simulation settings
POP_SIZE = 83      # cohort population size
SIM_LENGTH = 36    # length of simulation (months)
ALPHA = 0.05        # significance level for calculating confidence intervals
DISCOUNT = 0.03 # annual discount rate
DELTA_T_G = 1/12
DELTA_T_GC = 1/9

PSA_ON = True      # if probabilistic sensitivity analysis is on

# transition matrix
TRANS_MATRIX = [
    [0.1377,  0.1708,   0.6915],   # Progress Free
    [0,  0.2529,   0.7471],   # Progress
    [0,     0,    1],   # Death
    ]

TRANS_MATRIX_GC = [
    [0.2772,  0.1126, 0.6102],   # Progress Free
    [0,    0.4637,  0.5363],   # Progress
    [0,     0,    1],   # Death
    ]

# annual cost of each health state
MONTHLY_STATE_COST = [
    1501200,   # Progress Free
    1528813,   # Progress
    0    # Death
    ]

# annual health utility of each health state
MONTHLY_STATE_UTILITY = [
    0.690,   # Progress Free
    0.710,   # Progress
    0    # Death
    ]

# annual drug costs
G_COST = 43947
GC_COST = 44574

# treatment relative risk
Treatment_RR_G = 0.688
Treatment_RR_G_CI = 0.5508, 0.8252

Treatment_RR_GC = 0.707
Treatment_RR_GC_CI = 0.5678, 0.8461
