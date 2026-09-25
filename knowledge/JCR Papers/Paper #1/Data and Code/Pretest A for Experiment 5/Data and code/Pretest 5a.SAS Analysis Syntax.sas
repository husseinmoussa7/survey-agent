*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\pretest5a\Pretest 5a.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*RENAMING VARIABLES;
Data trains2;
 set trains0;
intent1=intent_to_buy_1;
intent2=intent_to_buy_5;
intent3=intent_to_buy_6;
intent4=intent_to_buy_7;
intent_ave=(intent1+intent2+intent3+intent4)/4;
concrete1=Q8_4;
concrete2=Q8_5;
concrete3=Q8_6;
concrete_ave=(concrete1+concrete2+concrete3)/3;
abstract1=Q8_1;
abstract2=Q8_2;
abstract3=Q8_3;
abstract_ave=(abstract1+abstract2+abstract3)/3;
if Q3_why1=1 then frame=1;
if Q4_how1=1 then frame=0;
run;



*Frequencies;
Proc freq data=trains2;
table frame gender;
run;

*Average age;
proc means data=trains2 mean stddev;
var age;
run;

*Descriptive statistics by conditions for key variables;
proc means data=trains2 mean median stddev stderr median min max;
class frame;
 var intent_ave concrete_ave abstract_ave ;
run;

*Alpha cronbach calculations;
proc corr data=trains2 nomiss alpha;
var intent1 intent2 intent3 intent4;
run;

proc corr data=trains2 nomiss alpha;
var concrete1 concrete2 concrete3;
run;

proc corr data=trains2 nomiss alpha;
var abstract1 abstract2 abstract3;
run;


*T-tests of key dependent measures;
proc ttest data=trains2;
   class frame;
   var intent_ave concrete_ave abstract_ave;
ods graphics off;
run;

