#include<fstream>
#include<getopt.h>
#include<iostream>
#include<stack>
#include"KmeanCuda.h"

#define LOG "log.txt"
#define BENCHLOG "BenchLog.txt"
#define MAXTHREADS 1024
#define MAXBLOCKS 65535
#define MAXSHAREDSIZE 

using namespace std;

int main(int argc, char** argv){
  
  struct Arguments args = parseArgs(argc,argv);	
  int nPoints = args.amount, nClusters = args.nclusters;
  float* data;
  struct Point *points;
  struct Cluster *clusters;
  ofstream logfile (LOG,ofstream::out);
  ofstream benchlog (BENCHLOG,ofstream::app);
  
  if (logfile == NULL){
		printf("Can't create outputfile %s\n",LOG);
		exit(1);
  }
  
   if (benchlog == NULL){
		printf("Can't create outputfile %s\n",BENCHLOG);
		exit(1);
  }
  
  //Allocate Memory 
  data = (float*) malloc(sizeof(float)*2*nPoints);
  points= (Point*) malloc(sizeof(Point)*nPoints);
  clusters = (Cluster*) malloc(sizeof(Cluster)*nClusters);
 
  //Perform Startup Tasks
  loadData(data,args.inputfile,nPoints*2);
  printf("Displaying first %d entries in Data:\n",10);
  displayData(data,10);
  DataToPoints(data,points,nPoints);
  printf("Displaying first %d points:\n", 4);
  displayPoints(points,4);
  initClusters(clusters,nClusters,nPoints);
  printf("Initialized %d Clusters with starting Positions:\n",nClusters);
  displayClusters(clusters,4);
 
  //Start the KMEANS Algorithm
  doKMeans(points,clusters,nPoints,nClusters,logfile,benchlog,0.01,args.mode);
    
  return 0;
}

void doKMeans(Point *points, Cluster *clusters, int nPoints, int nClusters, ofstream& logfile,ofstream& benchlog,float delta,int mode){
	stack<int> modes;
	modes.push(mode);
	struct Point *initClusterPositions;
	TIMERCREATE(KmeansSerial);
	TIMERCREATE(KmeansCUDA);
	//benchlog<<"####################################\n";
	//benchlog<<"######"<<"Points: "<<nPoints<<" Clusters: "<<nClusters<<" ######\n";
	benchlog<<nPoints<<"#"<<nClusters<<"#";
	printf("###  KMeans started ###\n");
	do{
	switch (modes.top()) {
		
		case 0:{
			
			//benchlog<<"\t SERIAL MODE\n";
			
			//Start the Serial Algorithm
			TIMERSTART(KmeansSerial);
			kMeansSerial(points,clusters,nPoints,nClusters,logfile,benchlog,delta);
			TIMERSTOP(KmeansSerial);
			
			//Display Result
			TIMERDISPLAY(KmeansSerial);
			displayClusters(clusters,nClusters);
			//benchlog<<"###### FINISHED IN: "<<timeKmeansSerial<<" ###### \n";
			benchlog<<timeKmeansSerial<<"#";
			modes.pop();
			break;
	
		}
		
		case 1:{
				
			//benchlog<<"\t CUDA MODE\n";
				
			//Create Device Structures
			struct Point *d_pointsCUDA;
			struct Cluster *d_clustersCUDA;
			
									
			//Allocate Device Memory
			cudaMalloc((void**)&d_pointsCUDA,sizeof(Point)*nPoints);
			checkCUDAError("malloc points");
			cudaMalloc((void**)&d_clustersCUDA,sizeof(Cluster)*nClusters);
			checkCUDAError("malloc points");
					
			//Init TIMERS
			TIMERSTART(KmeansCUDA)
			
			//Copy Data to Device		
			cudaMemcpy(d_pointsCUDA,points,sizeof(Point)*nPoints,cudaMemcpyHostToDevice);
			checkCUDAError("memcpy points");
			cudaMemcpy(d_clustersCUDA,clusters,sizeof(Cluster)*nClusters,cudaMemcpyHostToDevice);
			checkCUDAError("memcpy clusters");
		
			//Start the actual kMeans Algorithm		
			kMeansCUDA(d_pointsCUDA,d_clustersCUDA,nPoints,nClusters,delta,benchlog);
		
			//Copy Result to Host
			cudaMemcpy(points,d_pointsCUDA,sizeof(Point)*nPoints,cudaMemcpyDeviceToHost);
			checkCUDAError("memcpy points");
			cudaMemcpy(clusters,d_clustersCUDA,sizeof(Cluster)*nClusters,cudaMemcpyDeviceToHost);
			checkCUDAError("memcpy clusters");
			
			TIMERSTOP(KmeansCUDA)
			
			//Display Result
			for (int i = 0; i<nClusters;++i) printf("Cluster [%d] at Position: [%f,%f]\n",i,clusters[i].x,clusters[i].y);
			
			TIMERDISPLAY(KmeansCUDA)
			//benchlog<<"######  FINISHED IN: "<<timeKmeansCUDA<<" ######\n";
			benchlog<<timeKmeansCUDA<<"#";
			//Free Resources
			cudaFree(d_pointsCUDA);
			cudaFree(d_clustersCUDA);
			modes.pop();
			break;
		}
	
		case 2:{
			
			modes.pop();
			if (modes.empty()==true){
				
				initClusterPositions = (Point*) malloc(sizeof(Point)*nClusters);
				for (uint i = 0; i<nClusters;++i){
					initClusterPositions[i].x = clusters[i].x;
					initClusterPositions[i].y = clusters[i].y;
				}
				modes.push(-1);
				modes.push(2);
				modes.push(1);
			}
			else if(modes.top() == -1){
				modes.pop();
				for (uint i = 0; i<nClusters;++i){
					clusters[i].x = initClusterPositions[i].x;
					clusters[i].y = initClusterPositions[i].y;
					clusters[i].converged = false;
				}
				modes.push(0);
			}
					
			
		}
	
	
	}
	}while(!modes.empty());
	
	if(mode==2){
		TIMERCOMPARE(KmeansCUDA,KmeansSerial);
		//benchlog<<"\n###### SERIAL MODE: "<<timeKmeansSerial<<" ###### \n###### CUDA MODE: "<<timeKmeansCUDA<<" ###### \n###### SPEEDUP: "<<timeKmeansSerial/timeKmeansCUDA<<" ######\n";
		//benchlog<<"####################################\n";
		benchlog<<timeKmeansSerial/timeKmeansCUDA<<"\n";
	}
	
	//free(initClusterPositions);
	free(points);
	free(clusters);
	
	
}

void kMeansCUDA(Point *d_pointsCUDA, Cluster *d_clustersCUDA, int nPoints, int nClusters,float delta,ofstream& benchlog){
		
	int nBlocks,nThreads,converged,loopcount = 0;
	int *d_converged;

	cudaMalloc((void**)&d_converged,sizeof(int));
	checkCUDAError("malloc converged");
	
		
	nThreads = MAXTHREADS;
	nBlocks = ((nPoints/MAXTHREADS)<1?1:SDIV(nPoints,MAXTHREADS));
	
	if (nBlocks > MAXBLOCKS) nBlocks = MAXBLOCKS;
		
	printf("### CUDA MODE ###\n");
	printf("Starting the Algorithm with %d Blocks & %d Threads\n",nBlocks,nThreads);
		
	do{
		loopcount+=1;
		converged = 0;
		cudaMemcpy(d_converged,&converged,sizeof(int),cudaMemcpyHostToDevice);
		checkCUDAError("memcpy converged");
		//printf("### Loop : %d ###\n",loopcount);
		//logfile<<"###Loop "<<loopcount<<"\n";
		assignClosestClusterCUDA<<<nBlocks,nThreads>>>(d_pointsCUDA,d_clustersCUDA,nPoints,nClusters);
		cudaDeviceSynchronize();
		//printf("Here");
		checkCUDAError("Kernel Call");	
		//displayAssociatedPoints(clusters,nClusters);
		//logStatus(logfile,clusters,points,nClusters);
		findNewMeanCUDA<<<nBlocks,nThreads>>>(d_clustersCUDA,d_pointsCUDA,nClusters,nPoints,delta,d_converged);
		checkCUDAError("Kernel Call 2");
		//displayClusters(clusters,nClusters);
		cudaDeviceSynchronize();
		cudaMemcpy(&converged,d_converged,sizeof(int),cudaMemcpyDeviceToHost);
		checkCUDAError("memcpy converged");
		cudaDeviceSynchronize();
		//printf("%d",converged);
	}while (converged<nClusters);
		
	printf("### Converged in [%d] Loops ###\n",loopcount);
	//benchlog<<"Converged in ["<<loopcount<<"] Loops\n";
}

void kMeansSerial(Point *points, Cluster *clusters, int nPoints, int nClusters, ofstream& logfile,ofstream& benchlog,float delta){
	int loopcount = 0;
	printf("### SERIAL MODE ###\n");
	
	do{
			loopcount+=1;
			//printf("### Loop : %d ###\n",loopcount);
			logfile<<"###Loop "<<loopcount<<"\n";
			assignClosestCluster(points,clusters,nPoints,nClusters);
			//for (int i = 0;i<nPoints;++i) printf("(OUTSIDE) Point %d has Cluster %d\n",i,points[i].associatedCluster);
			//displayAssociatedPoints(clusters,nClusters);
			logStatus(logfile,clusters,points,nPoints,nClusters);
			findNewMean(clusters,points,nPoints,nClusters,delta);
			//displayClusters(clusters,nClusters);
		
		
	}while (checkConvergence(clusters,nClusters)==false);
	
	printf("### Converged in [%d] Loops ###\n",loopcount);
	logfile<<"Done\n";
	//benchlog<<"Converged in ["<<loopcount<<"] Loops\n";
	
}

__global__ void assignClosestClusterCUDA(Point *points,Cluster *clusters,int nPoints,int nClusters){
		
	int total_threads = blockDim.x*gridDim.x;
  	int globalTID= blockDim.x * blockIdx.x + threadIdx.x; 
	//uint globalTID = blockDim.x*blockIdx.x + threadIdx.x;
	
	float lowdistance=INFINITY,distance = 0;
	int ClusterID=0;
	
	for (int k = globalTID;k<nPoints;k+=total_threads){
		
		ClusterID = 0;
		distance = 0;
		lowdistance = INFINITY;
		for (uint i = 0;i<nClusters;i++){
			distance = (clusters[i].x - points[k].x)*(clusters[i].x - points[k].x) + (clusters[i].y - points[k].y)*(clusters[i].y - points[k].y);
			if (distance<lowdistance){
				//printf("%f\n",distance);
				lowdistance = distance;
				ClusterID= i;
			}
		}
	
	atomicExch(&points[k].associatedCluster,ClusterID);
	//printf("Associated point %d to Cluster %d\n",k,points[k].associatedCluster);
	atomicAdd(&clusters[ClusterID].nAssociatedPoints,1);
	}
	//printf("Done %d\n",globalTID);
}

__global__ void findNewMeanCUDA(Cluster *clusters,Point *points,int nClusters, int nPoints,float delta,int *d_converged){
	
	int total_threads = blockDim.x*gridDim.x;
  	int globalTID= blockDim.x * blockIdx.x + threadIdx.x; 	
	
	
	//uint globalTID = blockDim.x*blockIdx.x + threadIdx.x;
	float newX, newY;
	for (int k = globalTID;k<nClusters;k+=total_threads){
	//if (globalTID<nClusters){
		
		newX=0;
		newY=0;
		//printf("Cluster[%d] n associated Points: %d\n",k,clusters[k].nAssociatedPoints);
		if (clusters[k].nAssociatedPoints>0){
			for (uint i = 0;i<nPoints;i++){
				if(points[i].associatedCluster == k){
					newX += points[i].x;
					newY += points[i].y;
				}
			}
			newX = newX/clusters[k].nAssociatedPoints;
			newY = newY/clusters[k].nAssociatedPoints;
		}
		else{
			newX = clusters[k].x;
			newY = clusters[k].y;
		}
		//printf("new x: %f  old x: %f\n",newX,clusters[k].x);
		if(fabs(newX-clusters[k].x)<delta && fabs(newY-clusters[k].y)< delta) atomicAdd(&d_converged[0],1);
		//printf(" IN KERNEL %d\n",d_converged[0]);
		clusters[k].x = newX;
		clusters[k].y = newY;
		clusters[k].nAssociatedPoints=0;
		//printf("%f\n",clusters[k].x);
		
	
	}
}

/*void clustersToCUDA(Cluster *clusters, ClusterCUDA *clustersCUDA,int nClusters){
	for (uint i = 0;i<nClusters;++i){
		clustersCUDA[i].x = clusters[i].coords.x;
		clustersCUDA[i].y = clusters[i].coords.y;
		clustersCUDA[i].nAssociatedPoints = 0;
	}
}

void pointsToCUDA(Point *points, PointCUDA *pointsCUDA, int nPoints){
	for (uint i = 0; i<nPoints;i++){
		pointsCUDA[i].x = points[i].x;
		pointsCUDA[i].y = points[i].y;
		pointsCUDA[i].associatedCluster = 0;
	}
}*/

void assignClosestCluster(Point *points, Cluster *clusters,int nPoints, int nClusters){
	float lowdistance,distance;
	int ClusterID;
	#pragma omp parallel for private(ClusterID,lowdistance,distance)
	for (uint j = 0; j<nPoints;++j){
		ClusterID = 0;
		lowdistance = INFINITY;
		distance = 0;
		for (uint i = 0; i<nClusters;++i){
			distance = (clusters[i].x - points[j].x)*(clusters[i].x - points[j].x) + (clusters[i].y - points[j].y)*(clusters[i].y - points[j].y);
			if (distance<lowdistance){
				lowdistance = distance;
				ClusterID= i;
			}
			//printf("Distance: %f to Cluster: %d\n",distance,i);
		}
		#pragma omp critical 
		{
		points[j].associatedCluster = ClusterID;
		clusters[ClusterID].nAssociatedPoints +=1;
		}
		//printf("Assigned Point [%d] to Cluster [%d]\n",j,points[j].associatedCluster);
		
	}
	
}

void findNewMean(Cluster *clusters,Point *points, int nPoints, int nClusters,float delta){
	struct Point newCenter;
	//for (int i = 0;i<nPoints;++i) printf("(INSIDE)Point %d has Cluster %d\n",i,points[i].associatedCluster);
	//#pragma omp parallel for private(newCenter)
	for(uint i = 0; i<nClusters;++i){
		//printf("Cluster [%d] has [%d] Points associated\n",i,clusters[i].nAssociatedPoints);
		if (clusters[i].nAssociatedPoints>0){
			newCenter.x = 0;
			newCenter.y = 0;
			for (uint j = 0; j<nPoints; ++j){
				//printf("Cluster %d has %d Points ----- Point %d has Cluster %d\n",i,clusters[i].nAssociatedPoints,j,points[j].associatedCluster);
				if (points[j].associatedCluster == i){
					newCenter.x += points[j].x;
					newCenter.y += points[j].y;
				}
			}
			newCenter.x = newCenter.x/clusters[i].nAssociatedPoints;
			newCenter.y = newCenter.y/clusters[i].nAssociatedPoints;

		}
		else{
			newCenter.x = clusters[i].x;
			newCenter.y = clusters[i].y;
		}
		//printf("Cluster [%d]  new center x: %f old center x: %f\n",i,newCenter.x,clusters[i].x);
		//printf("distance : %f\n",fabs(newCenter.x - clusters[i].coords.x));
		if ((fabs(newCenter.x - clusters[i].x) < delta) && (fabs(newCenter.y - clusters[i].y) < delta)) clusters[i].converged = true;
		else clusters[i].converged = false;
	
		clusters[i].x = newCenter.x;
		clusters[i].y = newCenter.y;
		clusters[i].nAssociatedPoints = 0;
		
	}	
	
}

bool checkConvergence(Cluster *clusters,int nClusters){
	for (uint i = 0; i<nClusters; ++i){
		if(clusters[i].converged == false) return false;
	}
	return true;
}	

void logStatus(ofstream& logfile, Cluster *clusters, Point *points,int nPoints, int nClusters){
	for (uint i = 0; i<nClusters; ++i){
		logfile << "Cluster "<< i<<"\n";
		logfile<<"P "<< clusters[i].x << " " <<clusters[i].y<<"\n";
		for (uint j = 0; j<nPoints; ++j)
			if(points[j].associatedCluster == i)	logfile<<points[j].x<<" "<<points[j].y<<"\n";
		logfile<<"!\n";
		
	}		
}
	
void initClusters(Cluster *clusters, int nClusters, int nPoints){
	srand(time(NULL));
	for (uint i = 0; i<nClusters;++i){
		clusters[i].x = rand()%1000;
		clusters[i].y = rand()%1000;
		clusters[i].nAssociatedPoints = 0;
		clusters[i].converged = false;
	}
}

void displayClusters(Cluster *clusters, int nClusters){
	
	for (uint i = 0; i<nClusters; ++i) printf("Cluster [%d] at Position: [%f,%f] convergence: %d\n",i,clusters[i].x,clusters[i].y,clusters[i].converged);
 }

/*void displayAssociatedPoints(Cluster *clusters,int nClusters){
	for (uint i=0; i<nClusters; ++i){
		printf("Cluster %d has these Points associated with it:\n",i);
		for(uint j=0; j<clusters[i].nAssociatedPoints; ++j) printf("Point [%d]\n",clusters[i].associatedPoints[j]);
	}
}*/

void DataToPoints(float* data, Point *points,int N){
	for (uint i =0; i<N*2;i+=2) {
		points[(int)i/2].x = data[i];
		points[(int)i/2].y = data[i+1];
	}
}

void displayPoints(Point *points,int nPoints){
	for (uint i = 0;i<nPoints
	;++i) printf("Point [%d] at Position: [%f,%f]\n",i,points[i].x,points[i].y);
}

void loadData(float* data,char* fileName,int nElement){
  FILE* fin;
  fin = fopen(fileName,"r");
  if(fin==NULL)
  {
    printf("Can not open %s\n",fileName);
    exit(1);
  }
  
  fread(data,sizeof(float),nElement,fin);
  
  fclose(fin);
}

void displayData(float* data, int size){
  int i;
  for(i=0;i<size;++i) printf("%f ",data[i]);
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
		{"clusters",1,NULL,'c'},
		{"mode",1,NULL,'m'},
	};

		
	while((c=getopt_long(argc,argv,"n:i:c:m:",longOption,&optionIndex))!=-1){
		switch(c){
			case 'i':
					args.inputfile = strdup(optarg);
				    break;
			case 'n':
					args.amount = atoi(optarg);
					break;
			case 'c':
					args.nclusters = atoi(optarg);
					break;		
			case 'm':
					args.mode = atoi(optarg);
					break;				
			default:
					printf("Bad argument %c\n",c);
					exit(1);
		}
	}    
	
	return args;
}

void checkCUDAError(const char *msg){
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
