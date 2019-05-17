#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<getopt.h>


struct Arguments{
	int amount;
	char *inputfile;
};

Arguments parseArgs (int argc, char** argv);
void loadData(int* data,char* fileName,int nElement);
void displayData(int* data, int size);

int main(int argc, char** argv){
	
  struct Arguments args = parseArgs(argc,argv);	
  int* data;
  
  data = (int*) malloc(sizeof(int)*2*args.amount);
  
  loadData(data,args.inputfile,args.amount);
  
  displayData(data,args.amount);
  
  return 0;
}

void loadData(int* data,char* fileName,int nElement)
{
  FILE* fin;
  fin = fopen(fileName,"r");
  if(fin==NULL)
  {
    printf("Can not open %s\n",fileName);
    exit(1);
  }
  
  fread(data,sizeof(int),nElement,fin);
  
  fclose(fin);
}

void displayData(int* data, int size)
{
  int i;
  for(i=0;i<size;++i) printf("%d ",data[i]);
  printf("\n");
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
		{"input-file",1,NULL,'i'},
		{"amount",1,NULL,'n'},
	};

		
	while((c=getopt_long(argc,argv,"n:i:",longOption,&optionIndex))!=-1){
		switch(c){
			case 'i':
					args.inputfile = strdup(optarg);
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
