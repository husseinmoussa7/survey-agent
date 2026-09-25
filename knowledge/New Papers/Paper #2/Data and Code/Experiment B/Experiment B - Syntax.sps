* Encoding: UTF-8.

USE ALL.
COMPUTE filter_$=(Assigned_to_Condition = 1 & Finished = 1).
VARIABLE LABELS filter_$ 'Assigned_to_Condition = 1 & Finished = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

DESCRIPTIVES VARIABLES=Age
  /STATISTICS=MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=Gender
  /ORDER=ANALYSIS.

RECODE DV3 (7=1) (6=2) (5=3) (4=4) (3=5) (2=6) (1=7) INTO DV3_Reversed.
EXECUTE.

RELIABILITY
  /VARIABLES=DV1 DV2 DV3_Reversed
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA.

COMPUTE DemandTransparency=MEAN(DV1,DV2,DV3_Reversed).
EXECUTE.

T-TEST GROUPS=Agent(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=DemandTransparency
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).
