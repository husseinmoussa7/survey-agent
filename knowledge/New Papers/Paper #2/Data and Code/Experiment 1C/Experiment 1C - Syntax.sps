* Encoding: UTF-8.
USE ALL.
COMPUTE filter_$=(Assigned_to_Condition = 1).
VARIABLE LABELS filter_$ 'Assigned_to_Condition = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

CROSSTABS
  /TABLES=Condition BY Finished
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT ROW COLUMN TOTAL 
  /COUNT ROUND CELL.

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

UNIANOVA Understanding BY Agent Explain
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PLOT=PROFILE(Agent*Explain) TYPE=BAR ERRORBAR=SE(2) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent*Explain) COMPARE(Agent)
  /EMMEANS=TABLES(Agent*Explain) COMPARE(Explain)
  /PRINT ETASQ DESCRIPTIVE OPOWER
  /CRITERIA=ALPHA(.05)
  /DESIGN=Agent Explain Agent*Explain.

DO IF  (Agent =  - 1).
RECODE Explain (-1=0) (1=1) INTO Condition_Num.
END IF.
EXECUTE.
DO IF  (Agent =  1).
RECODE Explain (-1=2) (1=3) INTO Condition_Num.
END IF.
EXECUTE.

ONEWAY Understanding BY Condition_Num
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1  0 -1
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0  1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).







