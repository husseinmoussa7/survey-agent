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

CORRELATIONS
  /VARIABLES=Similarity_1 Similarity_2
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.

COMPUTE Perceived_Similarity=MEAN(Similarity_1,Similarity_2).
EXECUTE.

DO IF  (Dissimilarity = -1).
RECODE Agent (-1=1) (1=2) INTO Condition_Num.
END IF.
EXECUTE.
DO IF  (Dissimilarity = 1).
RECODE Agent (-1=3) (1=4) INTO Condition_Num.
END IF.
EXECUTE.

UNIANOVA Perceived_Similarity BY Agent Dissimilarity
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PLOT=PROFILE(Agent*Dissimilarity) TYPE=BAR ERRORBAR=SE(2) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent*Dissimilarity) COMPARE(Agent)
  /EMMEANS=TABLES(Agent*Dissimilarity) COMPARE(Dissimilarity)
  /PRINT ETASQ DESCRIPTIVE OPOWER
  /CRITERIA=ALPHA(.05)
  /DESIGN=Agent Dissimilarity Agent*Dissimilarity.

ONEWAY Perceived_Similarity BY Condition_Num
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1  0 -1
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0  1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

DESCRIPTIVES VARIABLES=Age 
  /STATISTICS=MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=Gender
  /ORDER=ANALYSIS.













