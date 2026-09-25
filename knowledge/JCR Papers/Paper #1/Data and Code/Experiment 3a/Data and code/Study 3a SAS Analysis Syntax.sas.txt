/*NEW LAPTOP COMPUTER NAME TRAIL*/

*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study3a\study 3a.Data.SAS.xlsx"
dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*renaming all variables;
Data trains1;
 set trains0;
 if condition=2 then speedX="slow";
 if condition=1 then speedX="fast";
 *recoding the BIF options where 1 = abstract, 0 = concrete;
 If biff1="A" then BIF1=1;
 If biff1="B" then BIF1=0;
 If biff3="A" then BIF3=1;
 If biff3="B" then BIF3=0;
 If biff4="A" then BIF4=1;
 If biff4="B" then BIF4=0;
 If biff5="A" then BIF5=1;
 If biff5="B" then BIF5=0;
 If biff7="A" then BIF7=1;
 If biff7="B" then BIF7=0;
 If biff8="A" then BIF8=1;
 If biff8="B" then BIF8=0;
 If biff10="A" then BIF10=1;
 If biff10="B" then BIF10=0;
 If biff13="A" then BIF13=1;
 If biff13="B" then BIF13=0;
 If biff14="A" then BIF14=1;
 If biff14="B" then BIF14=0;
 If biff16="A" then BIF16=1;
 If biff16="B" then BIF16=0;
 If biff20="A" then BIF20=1;
 If biff20="B" then BIF20=0;
 If biff23="A" then BIF23=1;
 If biff23="B" then BIF23=0;
 If biff24="A" then BIF24=1;
 If biff24="B" then BIF24=0;
*opposite codes;
 If biff2="A" then BIF2=0;
 If biff2="B" then BIF2=1;
 If biff6="A" then BIF6=0;
 If biff6="B" then BIF6=1;
 If biff9="A" then BIF9=0;
 If biff9="B" then BIF9=1;
 If biff11="A" then BIF11=0;
 If biff11="B" then BIF11=1;
 If biff12="A" then BIF12=0;
 If biff12="B" then BIF12=1;
 If biff15="A" then BIF15=0;
 If biff15="B" then BIF15=1;
 If biff17="A" then BIF17=0;
 If biff17="B" then BIF17=1;
 If biff18="A" then BIF18=0;
 If biff18="B" then BIF18=1;
 If biff19="A" then BIF19=0;
 If biff19="B" then BIF19=1;
 If biff21="A" then BIF21=0;
 If biff21="B" then BIF21=1;
 If biff22="A" then BIF22=0;
 If biff22="B" then BIF22=1;
 If biff25="A" then BIF25=0;
 If biff25="B" then BIF25=1;
bif_total=BIF1+BIF2+BIF3+BIF4+BIF5+BIF6+BIF7+BIF8+BIF9+BIF10+BIF11+BIF12+BIF13+BIF14+BIF15+BIF16+BIF17+BIF18+BIF19+BIF20+BIF21+BIF22+BIF23+BIF2+BIF25;
bif_ave=bif_total/25;
*recoding emotions related variables;
e1=emotion1;
e2=emotion2;
e3=8-emotion3;
e4=8-emotion4;
e5=8-emotion5;
e6=8-emotion6;
e7=8-emotion7;
e8=emotion8;
e9=emotion9;
e10=emotion10;
e11=8-emotion11;
emotions=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11)/11;
run;

*missing values for main DV;
proc freq data=trains2;
table bif_total;
run;

*remove missing values of BIF_total main DV;
Data trains2;
set trains2;
if BIF_total~=".";
run;

*Frequencies;
Proc freq data=trains2;
table speedX gender bif_total;
run;

*Average age;
proc means data=trains2 mean stddev;
var age;
run;

*Descriptive statistics by conditions for key variables;
proc means data=trains2 mean median stddev stderr median min max;
class speedX;
 var bif_total how_fast how_much_time moving_forwards easy_difficult_action emotions;
run;

*T-tests of key dependent measures;
proc ttest data=trains2;
   class speedX;
   var bif_total how_fast how_much_time moving_forwards easy_difficult_action emotions;
ods graphics off;
run;

*Alpha cronbach calculation for emotions measures;
proc corr data=trains2 nomiss alpha;
var e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11;
run;

