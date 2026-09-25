*READ IN DATA;
PROC IMPORT
DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study3c\Study 3c.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*RENAMING VARIABLES;
Data trains2;
 set trains0;
 if Q2_2=1 then speed=0;
 if Q2_2=1 then condition="slow";
 if Q3_2=1 then speed=1;
 if Q3_2=1 then condition="fast";
 if Q2_4=1 then pair1=0;
 if Q2_4=2 then pair1=1;
 if Q2_7=1 then pair2=0;
 if Q2_7=2 then pair2=1;
 if Q2_10=1 then pair3=0;
 if Q2_10=2 then pair3=1;
 pair_sum=pair1+pair2+pair3;
 p_speed=Q4_2;
 p_time=Q4_4;
 p_distance=Q4_6;
 ease=Q4_8;
 panas1=panas_1;
 panas2=panas_2;
 panas3=8-panas_3;
 panas4=8-panas_9;
 panas5=8-panas_10;
 panas6=8-panas_11;
 panas7=8-panas_12;
 panas8=panas_13;
 panas9=panas_14;
 panas10=panas_15;
 panas11=8-panas_16;
 panas_index=(panas1+panas2+panas3+panas4+panas5+panas6+panas7+panas8+panas9+panas10+panas11)/11;
 age=Q5_5;
 gender=Q5_6;
 device=Q5_7;
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
var pair_sum p_speed p_time p_distance ease panas_index;
run;

Title "T-tests";
proc ttest data=trains2;
   class condition;
   var pair_sum p_speed p_time p_distance ease panas_index;
ods graphics off;
run;

Title "Co-variate test";
proc glm data=trains2;
   class condition;
   model pair_sum=condition p_distance /ss3 effectsize;
run;

Title "PANAS Alpha";
proc corr data=trains2 nomiss alpha;
 var panas1 panas2 panas3 panas4 panas5 panas6 panas7 panas8 panas9 panas10 panas11;
 run;

proc sort;
by speed;
run;

*/Stack dataset for repeated measures analysis;
proc sort data= trains2;  by subnum; run;
Data Stacked;
 length subnum 8 replicate $ 6;
 keep  subnum choice replicate pair1 pair2 pair3 p_speed p_time p_distance ease panas_index age gender speed;
 Set trains2;
 choice=pair1;  replicate="pair1";  output;
 choice=pair2;  replicate="pair2";  output;
 choice=pair3;  replicate="pair3";   output;
 run;


Title "Pair Choice";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed/link=logit dist=binomial  ;
 repeated subject=subnum/type=un sorted;
 estimate "speed" speed 1 -1/ exp;
run;

Title "Perceived distance as covariate";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed p_distance/link=logit dist=binomial  ;
 repeated subject=subnum/type=un sorted;
 estimate "speed" speed 1 -1/ exp;
 estimate "p_distance" p_distance 1/exp;
run;

