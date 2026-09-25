
*1. READ IN DATA;
PROC IMPORT
 DATAFILE= "\\vmware-host\Shared Folders\forSAS\speed\R4\study2\Study 2.data.sas.xlsx"
 dbms=xlsx OUT=WORK.trains0 replace;
RUN;

*renaming all variables;
Data trains2;
	set trains0;
	if album=4 then album_b=1;
	if album=8 then album_b=0;
	if coffee=1 then coffee_b=1;
	if coffee=2 then coffee_b=0;
	if vid_game_d=1 then game_b=1;
	if vid_game_d=2 then game_b=0;
	if movie_desi=1 then movie_b=1;
	if movie_desi=2 then movie_b=0;
	if rest_desir=1 then food_b=1;
	if rest_desir=2 then food_b=0;
	if desk=2 then desk_b=1;
	if desk=1 then desk_b=0;
	total=album_b+coffee_b+game_b+movie_b+food_b+desk_b;
	d_destination=close_end;
	d_start=close_star;
	speed=Speed_perc;
	if in_against=1 then direction=0;
	if in_against=2 then direction=1;
run;

*frequencies of key DV;
proc freq data=trains2;
table speed;
run;

*Remove empties - for main DV;
Data trains2;
set trains2;
if speed~=".";
run;


*frequencies of key variables;
proc freq data=trains2;
table speed gender album_b coffee_b game_b movie_b food_b desk_b total direction;
run;

*descriptives of key variables;
proc means data=trains2 mean median stddev stderr median min max;
 var age speed d_destination d_start;
run;

*Create centered variables for continuous independent measures;
Data trains2;
set trains2;
speed_c=speed-5.2337662;
destination_c=d_destination-4.6233766;
start_c=d_start-4.9935065;
run;

*4. STACK THE DATA, ASCENDING BY SUBJECT ID;
proc sort data= trains2;  by subnum; run;
Data Stacked;
 length subnum 8 replicate $ 6;
 keep  subnum choice replicate speed speed_c destination_c start_c direction;
 Set trains2;
 choice=album_b;  replicate="album";  output;
 choice=coffee_b; replicate="coffee"; output;
 choice=game_b;   replicate="game";   output;
 choice=movie_b;  replicate="movie";  output;
 choice=food_b;   replicate="food";   output;
 choice=desk_b;   replicate="desk";   output; 
 run;

Title "Pair Choices";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed_c/link=logit dist=binomial;
 repeated subject=subnum/type=un sorted;
 estimate "speed_c" speed_c 1/ exp;
run;


Title "Pair Choices - direction as covariate";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed_c direction/link=logit dist=binomial  ;
 repeated subject=subnum/type=un sorted;
 estimate "speed_c" speed_c 1/ exp;
 estimate "direction" direction 1 / exp;
run;


Title "Pair Choices - start as covariate";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed_c start_c/link=logit dist=binomial  ;
 repeated subject=subnum/type=un sorted;
 estimate "speed_c" speed_c 1/ exp;
 estimate "start" start_c 1 / exp;
run;


Title "Pair Choices - destination as covariate";
proc genmod data=Stacked descending;
 class  subnum replicate;
 model choice =speed_c destination_c/link=logit dist=binomial  ;
 repeated subject=subnum/type=un sorted;
 estimate "speed_c" speed_c 1/ exp;
estimate "destination" destination_c 1 / exp;
run;

