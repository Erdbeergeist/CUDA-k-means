#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<getopt.h>

struct Arguments{
	int amount;
	char *outputfile;
};

Arguments parseArgs (int argc, char** argv);
void generatePair (int option = 0);

int main(int argc, char** argv){
	
	struct Arguments args = parseArgs(argc,argv);
	FILE* outfile;
	float* buffer;
	
	buffer = (float*) malloc(sizeof(float)*2*args.amount);
	
	printf("Generating %d integer pairs, storing in %s\n",args.amount,args.outputfile);
	
	srand(time(NULL));
	
	for (int i = 0;i<args.amount *2;++i)
		buffer[i] = rand()%100;
		
	outfile = fopen(args.outputfile,"w");
	
	if (outfile == NULL){
		printf("Can't create outputfile %s\n",args.outputfile);
		exit(1);
	}
	fwrite(buffer,sizeof(float),2*args.amount,outfile);
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
	};

		
	while((c=getopt_long(argc,argv,"n:o:",longOption,&optionIndex))!=-1){
		switch(c){
			case 'o':
					args.outputfile = strdup(optarg);
				    break;
			case 'n':
					args.amount = atoi(optarg);
					break;
			default:
					printf("Bad argument %c\n",c);
					exit(1);
		}
	}    
	
	return args;
	
}
