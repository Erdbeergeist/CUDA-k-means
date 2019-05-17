#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<getopt.h>




struct Arguments{
	int amount;
	int clamount;
	char *outputfile;
};

Arguments parseArgs (int argc, char** argv);
void generatePair (int option = 0);

int main(int argc, char** argv){
	
	struct Arguments args = parseArgs(argc,argv);
	FILE* outfile;
	int* buffer;
	
	int x,y;
	buffer = (int*) malloc(sizeof(int)*args.clamount*args.amount);
	
	printf("Generating %d integer pairs, storing in %s\n",args.amount,args.outputfile);
	
	srand(time(NULL));
	
	
	
	for (int i = 0;i<args.clamount;++i){
		x = rand()%100;
		y = rand()%100;
		for (int j = 0;j<args.amount;j+=2){
			buffer[i*args.amount +j] = x + (rand()%10) -5;
			buffer[i*args.amount +j+1] = y + (rand()%10) -5;
		}
	}
	
	for(int i = 0;i<args.clamount*args.amount;++i) printf("%d\n",buffer[i]);
	
	outfile = fopen(args.outputfile,"w");
	
	if (outfile == NULL){
		printf("Can't create outputfile %s\n",args.outputfile);
		exit(1);
	}
	fwrite(buffer,sizeof(int),args.clamount*args.amount,outfile);
	fclose(outfile);
	
	printf("Created file %s successfully\n",args.outputfile);
	
	free(buffer);
	
	return 0;
}

Arguments parseArgs(int argc, char** argv){
	
	char c;
    int optionIndex = 0;
	
	struct Arguments args;
	
	if (argc<4) {
		printf("To few Arguments!\n");
		exit(1);
	}
	
	struct option longOption[]={
		{"output-file",1,NULL,'o'},
		{"amount",1,NULL,'n'},
		{"clamount",1,NULL,'c'},
	};

		
	while((c=getopt_long(argc,argv,"n:o:c:",longOption,&optionIndex))!=-1){
		switch(c){
			case 'o':
					args.outputfile = strdup(optarg);
				    break;
			case 'n':
					args.amount = atoi(optarg);
					break;
			case 'c':
					args.clamount = atoi(optarg);
					break;		
			default:
					printf("Bad argument %c\n",c);
					exit(1);
		}
	}    
	
	return args;
	
}
