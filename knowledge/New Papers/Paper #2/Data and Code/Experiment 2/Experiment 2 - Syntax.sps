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


DO IF  (SelfUnderstanding =  - 1).
RECODE Agent (-1=1) (1=2) INTO Condition_Num.
END IF.
EXECUTE.
DO IF  (SelfUnderstanding =  1).
RECODE Agent (-1=3) (1=4) INTO Condition_Num.
END IF.
EXECUTE.

UNIANOVA Self BY SelfUnderstanding Agent 
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PLOT=PROFILE(Agent*SelfUnderstanding) TYPE=BAR ERRORBAR=SE(1) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent) 
  /EMMEANS=TABLES(SelfUnderstanding) 
  /EMMEANS=TABLES(SelfUnderstanding*Agent) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER
  /CRITERIA=ALPHA(.05)
  /DESIGN=Agent SelfUnderstanding SelfUnderstanding*Agent.

GLM Understanding_1 Understanding_2 BY Agent SelfUnderstanding
  /WSFACTOR=Understanding 2 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(Agent*Understanding*SelfUnderstanding) TYPE=BAR ERRORBAR=SE(1) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent*SelfUnderstanding*Understanding) Compare (Understanding)
   /EMMEANS=TABLES(Agent*SelfUnderstanding*Understanding) Compare (Agent) 
    /EMMEANS=TABLES(Agent*SelfUnderstanding*Understanding) Compare (SelfUnderstanding)
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Understanding 
  /DESIGN=Agent SelfUnderstanding Agent*SelfUnderstanding.

ONEWAY Understanding_1 BY Condition_Num
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0 1 -1
  /CONTRAST=1 0 -1 0
  /CONTRAST=0 1 0 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES HOMOGENEITY 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

ONEWAY Understanding_2 BY Condition_Num
  /CONTRAST=1 -1 0 0
  /CONTRAST=0 0 1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES HOMOGENEITY 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).


COMPUTE IOED=Understanding_1 - Understanding_2.
EXECUTE.

UNIANOVA IOED BY SelfUnderstanding Agent 
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PLOT=PROFILE(Agent*SelfUnderstanding) TYPE=BAR ERRORBAR=SE(1) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent) 
  /EMMEANS=TABLES(SelfUnderstanding) 
  /EMMEANS=TABLES(SelfUnderstanding*Agent) Compare (Agent)
    /EMMEANS=TABLES(SelfUnderstanding*Agent) Compare (SelfUnderstanding)
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER
  /CRITERIA=ALPHA(.05)
  /DESIGN=Agent SelfUnderstanding SelfUnderstanding*Agent.

ONEWAY IOED BY Condition_Num
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0  0 1 -1
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1 0 -1 
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES HOMOGENEITY 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

USE ALL.
COMPUTE filter_$=(Assigned_to_Condition = 1 & Finished = 1 & SelfUnderstanding = -1).
VARIABLE LABELS filter_$ 'Assigned_to_Condition = 1 & Finished = 1 & SelfUnderstanding = -1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

USE ALL.
COMPUTE filter_$=(Assigned_to_Condition = 1 & Finished = 1 & SelfUnderstanding = 1).
VARIABLE LABELS filter_$ 'Assigned_to_Condition = 1 & Finished = 1 & SelfUnderstanding = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.


SORT CASES  BY Agent.
SPLIT FILE SEPARATE BY Agent.

CORRELATIONS
  /VARIABLES=Self Understanding_1 
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.

T-TEST PAIRS=Understanding_1 WITH Understanding_2 (PAIRED)
  /ES DISPLAY(TRUE) STANDARDIZER(SD)
  /CRITERIA=CI(.9500)
  /MISSING=ANALYSIS.

DESCRIPTIVES VARIABLES=Age
  /STATISTICS=MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=Gender
  /ORDER=ANALYSIS.

*** We compared correlation coefficients using the following calculator: 
    https://www.danielsoper.com/statcalc/calculator.aspx?id=104 ***
