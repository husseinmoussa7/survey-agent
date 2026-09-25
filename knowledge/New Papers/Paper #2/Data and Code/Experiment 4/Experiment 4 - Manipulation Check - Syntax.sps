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

ONEWAY Perceived_Similarity BY Condition_Number
  /CONTRAST=1 -1 0 
  /CONTRAST=1 0 -1 
  /CONTRAST=0 1 -1 
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

DESCRIPTIVES VARIABLES=Age 
  /STATISTICS=MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=Gender
  /ORDER=ANALYSIS.














