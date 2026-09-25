*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\pretest5b\Pretest 5b.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;


*RENAMING VARIABLES;
Data trains2;
 set trains0;
if fast1=1 then video=1;
if fast2=1 then video=2;
if slow1=1 then video=3;
if slow3=1 then video=4;
if slow_5=1 then video=5;
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
if apple="apple" then acheck=1;
if apple="Apple" then acheck=1;
speed_ave=(speed+speed_feel)/2;
run;

*Frequencies;
Proc freq data=trains2;
table video gender;
run;

*Average age;
proc means data=trains2 mean stddev;
var age;
run;

*Descriptive statistics by conditions for key variables;
proc means data=trains2 mean median stddev stderr median min max;
class video;
 var panas_index speed_feel speed speed_ave;
run;

*Alpha cronbach calculations;
proc corr data=trains2 nomiss alpha;
var inspired alert afraid upset nervous scared distressed excited enthusiastic determined bored;
run;

*Alpha cronbach calculations;
proc corr data=trains2 nomiss alpha;
var speed speed_feel;
run;

proc glm data=trains2;
class video;
model panas_index=video;
lsmeans video / pdiff;
run;


proc glm data=trains2;
class video;
model speed_ave=video;
lsmeans video / pdiff;
run;

