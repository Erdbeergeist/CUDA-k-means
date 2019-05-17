#ifndef KMEANCUDA_H
#define KMEANCUDA_H

#define SDIV(x,y)(((x)+(y)-1)/(y))

#define TIMERCREATE(label)													 \
		cudaEvent_t start##label, stop##label;                               \
        float time##label;         

#define TIMERSTART(label)                                                    \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);      
         
#define TIMERDISPLAY(label) std::cout << time##label << " ms (" << #label << ")" << std::endl;

#define TIMERCOMPARE(label,label2) printf("CUDA TIME: %f  vs SERIAL TIME: %f ---> SPEEDUP: %f\n",time##label,time##label2,time##label2/time##label);

#define TIMERWRITE(label) time##label;

typedef unsigned int uint;
typedef unsigned char uchar;

struct Arguments{
	int amount;
	int nclusters;
	char *inputfile;
	int mode;
};

struct Point{
	float x;
	float y;
	int associatedCluster;
};

struct Cluster{
	float x;
	float y;
	int nAssociatedPoints;
	bool converged;
};

Arguments parseArgs (int argc, char** argv);
void loadData(float* data,char* fileName,int nElement);
void displayData(float* data, int size);
void DataToPoints(float* data, Point *points,int N);
void initClusters(Cluster *clusters, int nClusters, int nPoints);
void displayClusters(Cluster *clusters, int nClusters);
void displayPoints(Point *points, int nPoints);
void doKMeans(Point *points, Cluster *clusters,int nPoints, int nClusters,std::ofstream& logfile,std::ofstream& benchlog,float delta,int mode);
void assignClosestCluster(Point *points,Cluster *clusters,int nPoints,int nClusters);
void displayAssociatedPoints(Cluster *clusters, int nClusters);
void findNewMean(Cluster *clusters,Point *points,int nPoints,int nClusters,float delta);
bool checkConvergence(Cluster *clusters,int nClusters);
void logStatus(std::ofstream& logfile,Cluster *clusters,Point *points,int nPoints,int nClusters);
void kMeansSerial(Point *points, Cluster *clusters, int nPoints, int nClusters, std::ofstream& logfile,std::ofstream& benchlog,float delta);



// CUDA FUNCTIONS //
void checkCUDAError(const char *msg);
void pointsToCUDA(Point *points, Point *pointsCUDA,int nPoints);
void clustersToCUDA(Cluster *clusters, Cluster *clustersCUDA, int nClusters);
void kMeansCUDA(Point *d_pointsCUDA, Cluster *d_clustersCUDA, int nPoints, int nClusters,float delta,std::ofstream& benchlog);


// CUDA DEVICE FUNCTIONS //
__global__ void assignClosestClusterCUDA(Point *point,Cluster *clusters,int nPoints,int nClusters);
__global__ void findNewMeanCUDA(Cluster *clusters,Point *points, int nClusters, int nPoints,float delta,int *d_converged);


#endif
