*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study5\Study 5.Data.SAS.xlsx"
 dbms=xlsx OUT=WORK.trains0 replace;
RUN;


*renaming variables;
Data trains1;
 set trains0;
 if speed=0 then speedX="slow";
 if speed=1 then speedX="fast";
 if CLT=0 then CLTX="concrete";
 if CLT=1 then CLTX="abstract";
 if speed=0 then speed_e=-1;
 if speed=1 then speed_e=1;
 if clt=0 then clt_e=-1;
 if clt=1 then clt_e=1;
 interaction=speed*CLT;
 interaction_e=speed_e*CLT_e;
 *creation of contrast analysis variables;
 b_clt_e=clt_e+1;
 a_clt_e=clt_e-1;
 b_interaction=b_clt_e*speed_e;
 a_interaction=a_clt_e*speed_e;
 run;

Title "Frequencies of Key DVs";
proc freq data=trains1;
table speed CLT DVThruPlays DV3sec DVLinkClicks;
run;

Title "Key Means";
proc means data=trains1 mean median stddev stderr median min max;
 var DVThruPlays DV3sec DVLinkClicks;
run;

proc sort;
By speedX cltx;
run;

Title "Key Means by condition";
proc means data=trains1 mean median stddev stderr median min max;
class speedX cltX;
var DVThruPlays DV3sec DVLinkClicks;
run;


proc sort;
By speedX;
run;

Title "Key Means by speed";
proc means data=trains1 mean median stddev stderr median min max;
class speedX;
var DVThruPlays DV3sec DVLinkClicks;
run;

proc sort;
By CLTX;
run;

Title "Key Means by speed";
proc means data=trains1 mean median stddev stderr median min max;
class CLTX;
var DVThruPlays DV3sec DVLinkClicks;
run;


Title "3 Second Play Through - Logit Analysis with effect coded variables";
proc genmod data=trains1 descending;
 model DV3sec =speed_e CLT_e interaction_e/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" CLT_e 1/ exp;
 estimate "interaction" interaction_e 1/ exp;
run;


Title "3 Second Play Through - Contrast analysis in concrete";
proc genmod data=trains1 descending;
 model DV3sec =speed_e b_CLT_e b_interaction/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" b_CLT_e 1/ exp;
 estimate "interaction" b_interaction 1/ exp;
run;


Title "3 Second Play Through - Contrast analysis in abstract";
proc genmod data=trains1 descending;
 model DV3sec =speed_e a_CLT_e a_interaction/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" a_CLT_e 1/ exp;
 estimate "interaction" a_interaction 1/ exp;
run;



Title "ThruPlay - Logit Analysis with effect coded variables";
proc genmod data=trains1 descending;
 model DVThruplays =speed_e CLT_e interaction_e/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" CLT_e 1/ exp;
 estimate "interaction" interaction_e 1/ exp;
run;


Title "ThruPlay - Contrast analysis in concrete";
proc genmod data=trains1 descending;
 model DVThruplays =speed_e b_CLT_e b_interaction/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" b_CLT_e 1/ exp;
 estimate "interaction" b_interaction 1/ exp;
run;


Title "ThruPlay - Contrast analysis in abstract";
proc genmod data=trains1 descending;
 model DVThruplays =speed_e a_CLT_e a_interaction/link=logit dist=binomial;
 estimate "speed" speed_e 1/ exp;
 estimate "CLT" a_CLT_e 1/ exp;
 estimate "interaction" a_interaction 1/ exp;
run;
