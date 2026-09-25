*READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study1\Study 1.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*RENAMING VARIABLES;
Data trains2;
 set trains0;
if fast_train=1 then do;
speed=1;
condition="fast";
jobs1=relative_appeal;
jobs2=likely_to_apply;
jobs_ave=(jobs1+jobs2)/2;
end;
if slow_train=1 then do;
speed=0;
condition="slow";
jobs1=relative_appeal_1;
jobs2=likely_to_apply_1;
jobs_ave=(jobs1+jobs2)/2;
end;
p_speed=perceived____speed2;
p_time=perceivedtime_passed;
p_distance=perceived_distance;
 ease=diffuculty;
inspired=panas_1;
alert=panas_2;
afraid=8-panas_3;
upset=8-panas_9;
nervous=8-panas_10;
scared=8-panas_11;
distressed=8-panas_12;
excited=panas_13;
enthusiastic=panas_14;
determined=panas_15;
bored=8-panas_16;
panas_index=(inspired+alert+afraid+upset+nervous+scared+distressed+excited+enthusiastic+determined+bored)/11;
run;

Title "Frequencies";
proc freq data=trains2;
table condition gender;
run;

Title "age";
proc means data=trains2 mean median stddev stderr median min max;
 var age;
run;

Title "Key means by condition";
proc means data=trains2 mean median stddev stderr median min max;
class condition;
var jobs1 jobs2 jobs_ave p_speed p_time p_distance ease panas_index ;
run;

Title "T-tests";
proc ttest data=trains2;
   class condition;
   var p_speed p_time p_distance ease panas_index;
ods graphics off;
run;

Title "PANAS Alpha";
proc corr data=trains2 nomiss alpha;
 var inspired alert afraid upset nervous scared distressed excited enthusiastic determined bored;
 run;

*/Stack dataset for repeated measures analysis;
proc sort data= trains2;  by subnum; run;
Data Stacked;
 length subnum 8 replicate $ 6 ;
 keep  subnum choice replicate speed p_speed p_time p_distance ease panas_index;
 Set trains2;
 choice=jobs1; replicate="1"; output;
 choice=jobs2; replicate="2"; output;
  run;


Title "Mixed ANOVA - job preference measures";
proc mixed data=Stacked;
class speed replicate subnum;
model choice=speed;
repeated /group=speed subject=subnum type=un;
LSMeans speed;
run;


Title "Mixed ANOVA - job preference measures with distance covariate";
proc mixed data=Stacked;
class speed replicate subnum;
model choice=speed p_distance;
repeated /group=speed subject=subnum type=un;
LSMeans speed;
run;

Title "Mixed ANOVA - job preference measures with panas covariate";
proc mixed data=Stacked;
class speed replicate subnum;
model choice=speed panas_index;
repeated /group=speed subject=subnum type=un;
LSMeans speed;
run;
