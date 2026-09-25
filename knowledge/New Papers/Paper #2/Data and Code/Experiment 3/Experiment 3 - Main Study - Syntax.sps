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

DESCRIPTIVES VARIABLES=Age Age_corrected
  /STATISTICS=MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=Gender
  /ORDER=ANALYSIS.

GLM Understanding_1 Understanding_2 BY Agent Dissimilarity
  /WSFACTOR=Understanding 2 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(Agent*Dissimilarity*Understanding) TYPE=BAR ERRORBAR=SE(1) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent*Dissimilarity*Understanding) Compare (Agent)
  /EMMEANS=TABLES(Agent*Dissimilarity*Understanding) Compare (Dissimilarity)
  /EMMEANS=TABLES(Agent*Dissimilarity*Understanding) Compare (Understanding)
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Understanding 
  /DESIGN=Agent Dissimilarity Agent*Dissimilarity.

ONEWAY Understanding_1 BY Condition_Num
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1  0 -1
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0  1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

ONEWAY Understanding_2 BY Condition_Num
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1  0 -1
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0  1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

COMPUTE IOED=Understanding_1 - Understanding_2.
EXECUTE.

UNIANOVA IOED BY Agent Dissimilarity
  /METHOD=SSTYPE(3)
  /INTERCEPT=INCLUDE
  /PLOT=PROFILE(Agent*Dissimilarity) TYPE=BAR ERRORBAR=SE(2) MEANREFERENCE=NO
  /EMMEANS=TABLES(Agent*Dissimilarity) COMPARE(Agent)
  /EMMEANS=TABLES(Agent*Dissimilarity) COMPARE(Dissimilarity)
  /PRINT ETASQ DESCRIPTIVE OPOWER
  /CRITERIA=ALPHA(.05)
  /DESIGN=Agent Dissimilarity Agent*Dissimilarity.

ONEWAY IOED BY Condition_Num
  /CONTRAST=1 0 -1 0 
  /CONTRAST=0 1  0 -1
  /CONTRAST=1 -1 0 0 
  /CONTRAST=0 0  1 -1
  /ES=OVERALL CONTRAST(POOLED)
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).









