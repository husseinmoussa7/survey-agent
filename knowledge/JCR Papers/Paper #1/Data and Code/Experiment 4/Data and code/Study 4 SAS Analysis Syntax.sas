*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study4\Study 4.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*RENAMING VARIABLES;
Data trains2;
 set trains0;
screen=browser_stuff_4_text;
p_speed=speed;*recode into perceived speed variable;
if condition='slow new' then do;
 speed=0; speedx='slow';
 end;
 if condition='fast' then do;
 speed=1; speedx='fast';
 end;
 if MOD='CONSISTENT' then do;
 moderation=0; moderationx='consistent';
 end;
 if MOD='INCONSISTENT' then do;
 moderation=1; moderationx='inconsistent';
 end;
 if cs_intro=1 then buyit=cs_buyit;
 if is_intro=1 then buyit=is_buyit;
 if cf_intro=1 then buyit=cf_buyit;
 if if_intro=1 then buyit=if_buyit;
 if pair1=1 then p1=0;
 if pair1=2 then p1=1;
 if pair2=1 then p2=0;
 if pair2=2 then p2=1;
 if pair3=1 then p3=0;
 if pair3=2 then p3=1;
 pair_sum=p1+p2+p3;
 dv_rt=Q134_3;
 jobs1=jobs_apeal;
 jobs2=jobs_apply;
run;

*Remove people who fail attention check;
Data trains2;
set trains2;
if intro=".";
run;

*Frequencies of key variables;
proc freq data=trains2;
table gender speedx*moderationx speedX moderationX;
run;

proc means data=trains2 mean median stddev stderr median min max;
var age;
run;

proc sort;
by speedx;
run;
Title "Means by speed";
proc means data=trains2 mean median stddev stderr median min max;
class speedx;
 var jobs1 jobs2 pair_sum p1 p2 p3 p_speed buyit;
run;

proc sort;
by moderationx;
run;
Title "Means by association";
proc means data=trains2 mean median stddev stderr median min max;
class moderationx;
 var jobs1 jobs2 pair_sum p_speed p1 p2 p3 buyit;
run;

proc sort;
by speedx moderationx;
run;
Title "Means by both conditions";
proc means data=trains2 mean median stddev stderr median min max;
class speedx moderationx;
 var jobs1 jobs2 pair_sum p_speed p1 p2 p3 buyit time_passed difficulty;
run;

Title "Speed Manipulation Check";
proc glm data=trains2;
class speedx moderationx;
model p_speed=speedx|moderationx/effectsize;
lsmeans speedx;
ODS GRAPHICS OFF;
run;


Title "Associations manipulation check";
proc glm data=trains2;
class speedx moderationx;
model pair_sum=speedx|moderationx/effectsize;
lsmeans speedx*moderationx/slice=speedx slice=moderationx;
lsmeans speedx*moderationx/pdiff;
ODS GRAPHICS OFF;
run;

Title "Believability of associations manipulation check";
proc glm data=trains2;
class speedx moderationx;
model buyit=speedx|moderationx/effectsize;
ODS GRAPHICS OFF;
run;

Title "Perceived Time alternative explanation check";
proc glm data=trains2;
class speedx moderationx;
model time_passed=speedx|moderationx/effectsize;
lsmeans speedx*moderationx/slice=speedx slice=moderationx;
lsmeans speedx*moderationx/pdiff;
ODS GRAPHICS OFF;
run;

Title "Perceived ease alternative explanation check";
proc glm data=trains2;
class speedx moderationx;
model difficulty=speedx|moderationx/effectsize;
lsmeans speedx*moderationx/slice=speedx slice=moderationx;
lsmeans speedx*moderationx/pdiff;
ODS GRAPHICS OFF;
run;


*Stack the dataset;
proc sort data= trains2;  by subnum; run;
Data Stacked;
 length subnum 8 replicate $ 6 ;
 keep  subnum choice replicate pair_sum p1 p2 p3 speedX moderationX;
 Set trains2;
 choice=jobs1; replicate="1"; output;
 choice=jobs2; replicate="2"; output;
  run;

Title "Job Preference Dependent measures";
proc mixed data=Stacked;
class speedX moderationX subnum;
model choice= speedX moderationX speedX*moderationX;
repeated / group=speedX*moderationX subject=subnum type=un;
lsmeans speedx*moderationx/slice=speedx slice=moderationx;
lsmeans speedx*moderationx/pdiff;
run;
