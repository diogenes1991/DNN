#include <iostream>
#include <vector>
#include <random>
#include <pthread.h>
#include <thread>
#include <fstream>

#include <sched.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <random> 


#define USEGSL 1

// #ifdef USEGSL
#include <gsl/gsl_linalg.h>
// #undef solver
// #undef matrix 
// #undef vector
// #define invert gsl_linalg_SV_solve
// #define vector gsl_vector
// #define matrix gsl_matrix
// #endif


#define DEBUG 0

thread_local std::mt19937 rng(0);

double RANDOM(double a, double b){
    return std::uniform_real_distribution<double>{a,b}(rng);
}

class Matrix{

    gsl_matrix * m;

    public:

        Matrix(size_t n1, size_t n2){
            m = gsl_matrix_alloc(n1,n2);
        }
        ~Matrix(){
            gsl_matrix_free(m);
        }
        void Set(size_t i, size_t j, double mij){
            gsl_matrix_set(m,i,j,mij);
        }
        double Get(size_t i, size_t j){return gsl_matrix_get(m,i,j);}
};

template<class T>
class NodeT{

    int N;
    T* Weights;

    public:

        NodeT(int n){
            N = n;
            Weights = new T[N];
            for(int i=0;i<N;i++){
                Weights[i] = T(1)/(T(N));
            }
        }
    
        ~NodeT(){
            delete [] Weights;
        }

        int GetN(){return N;}

        T GetWeight(int i){return Weights[i];}
        void SetWeights(T* W){for(int i=0;i<N;i++)Weights[i]=W[i];}

        void Show(std::string name = ""){
            if(name!="")std::cout<<"Node "<<name<<" ";
            std::cout<<"Weights = {";
            for(int i=0;i<N;i++){
                std::cout<<Weights[i]<<" ";
            }
            std::cout<<"}"<<std::endl;
        }

        void Evaluate(T*INPUT, T* RVAL){
            *RVAL = 0;
            T AUX = 0;
            for(int i=0;i<N;i++)AUX += Weights[i]*INPUT[i];
            *RVAL = AUX /(1+std::abs(AUX));
        }

        void EvaluateDer(T* INPUT, T* RVAL){
            *RVAL = 0;
            T AUX = 0;
            for(int i=0;i<N;i++)AUX += Weights[i]*INPUT[i];
            *RVAL = T(1)/((1+std::abs(AUX))*(1+std::abs(AUX)));
        }
};

template<class T>
class LayerT{

    int N,M;
    NodeT<T>** Nodes;

    public:
        
        LayerT(int n, int m){
            N = n;
            M = m;
            Nodes = new NodeT<T>*[N];
            for(int i=0;i<N;i++){
                Nodes[i] = new NodeT<T>(M);
            }
        }

        ~LayerT(){
            for(int i=0;i<N;i++) delete Nodes[i];
            delete [] Nodes;
        }

        int GetN(){return N;}
        int GetM(){return M;}
        T GetWeight(int Node, int Wgt){return Nodes[Node]->GetWeight(Wgt);}

        void Show(std::string Name = ""){
            if(Name!="")std::cout<<"Layer "<<Name<<":"<<std::endl;
            for(int i=0;i<N;i++){
                Nodes[i]->Show(std::to_string(i));
            }
        }

        void Evaluate(T* INPUT, T* RVAL){
            T AUX;
            for(int i=0;i<N;i++){
                AUX = 0;
                Evaluate(i,INPUT,&AUX);
                RVAL[i] = AUX;
            }
        }

        void EvaluateDer(T* INPUT, T* RVAL){
            T AUX;
            for(int i=0;i<N;i++){
                AUX = 0;
                EvaluateDer(i,INPUT,&AUX);
                RVAL[i] = AUX;
            }
        }

        void SetWeights(int n, T* W){Nodes[n]->SetWeights(W);}

    private:

        NodeT<T>* GetNode(int i){return Nodes[i];}
        
        void Evaluate(int n, T* INPUT, T* RVAL){
            *RVAL = 0;
            Nodes[n]->Evaluate(INPUT,RVAL);
        }

        void EvaluateDer(int n, T* INPUT, T* RVAL){
            *RVAL = 0;
            Nodes[n]->EvaluateDer(INPUT,RVAL);
        }
};

template <class T>
class NetworkT{

    int N;
    int* Specs;
    LayerT<T>** Layers;

    public:

        NetworkT(int n, int* Spec){
            N = n;
            Layers = new LayerT<T>*[N];
            Specs  = new int[N+1]; 
            for(int i=0;i<N;i++){
                Specs[i]   = Spec[i];
                Specs[i+1] = Spec[i+1];
                Layers[i] = new LayerT<T>(Spec[i+1],Spec[i]);
            }
        }

        void Show(bool deep = false){
            std::cout<<"Neural Network with the following Architechture:"<<std::endl;
            std::cout<<"In["<<Specs[0]<<"]->";
            for(int i=1;i<N+1;i++){
                std::cout<<"Layer_"<<i<<"("<<Specs[i-1]<<","<<Specs[i]<<")->";
            }
            std::cout<<"Ou["<<Specs[N]<<"]"<<std::endl;
            if(deep){
                for(int i=1;i<N+1;i++) Layers[i-1]->Show(std::to_string(i));
            }
        }

        void SetWeights(int Layer, int Node, T* Wgt){Layers[Layer]->SetWeights(Node,Wgt);}
        T GetWeight(int Layer, int Node, int Wgt){return Layers[Layer]->GetWeight(Node,Wgt);}

        ~NetworkT(){
            for(int i=0;i<N;i++) delete Layers[i];
            delete [] Layers;
            delete [] Specs;
        }

        void EvaluateHA(T* INPUT, T* OUTPUT){ //Heap Allocated
            T *IN,*OU;
            
            OU = new T[Specs[0]];
            for(int j=0;j<Specs[0];j++)OU[j]=INPUT[j];
            
            for(int i=0;i<N;i++){

                IN = new T[Specs[i]];
                for(int j=0;j<Specs[i];j++)IN[j]=OU[j];
                
                #if DEBUG
                    std::cout<<"The inputs to this layer are: {";
                    for(int j=0;j<Specs[i];j++)std::cout<<IN[j]<<" ";
                    std::cout<<"}\n";
                #endif
                
                delete OU;
                OU = new T[Specs[i+1]];

                Layers[i]->Evaluate(IN,OU);

                #if DEBUG
                    std::cout<<"The outputs of this layer are: {";
                    for(int j=0;j<Specs[i+1];j++)std::cout<<OU[j]<<" ";
                    std::cout<<"}\n";
                #endif
                
                delete IN;  
            }

            for(int j=0;j<Specs[N];j++)OUTPUT[j]=OU[j];
            
            delete OU;
        }

        void Evaluate(T* INPUT, T* OUTPUT){ // Stack Allocated
            int MAX = 0;
            for(int i=0;i<N;i++)if(MAX<Specs[i])MAX=Specs[i];
            T IN[MAX],OU[MAX];
            
            for(int j=0;j<Specs[0];j++)OU[j]=INPUT[j];
            
            for(int i=0;i<N;i++){

                for(int j=0;j<Specs[i];j++)IN[j]=OU[j];
                
                #if DEBUG
                    std::cout<<"The inputs to this layer are: {";
                    for(int j=0;j<Specs[i];j++)std::cout<<IN[j]<<" ";
                    std::cout<<"}\n";
                #endif
                
                Layers[i]->Evaluate(IN,OU);

                #if DEBUG
                    std::cout<<"The outputs of this layer are: {";
                    for(int j=0;j<Specs[i+1];j++)std::cout<<OU[j]<<" ";
                    std::cout<<"}\n";
                #endif
            }

            for(int j=0;j<Specs[N];j++)OUTPUT[j]=OU[j];
        }

        void EvaluateDer(T* INPUT, T* OUTPUT){ // Stack Allocated
            int MAX = 0;
            for(int i=0;i<N;i++)if(MAX<Specs[i])MAX=Specs[i];
            T IN[MAX],OU[MAX];
            
            for(int j=0;j<Specs[0];j++)OU[j]=INPUT[j];
            
            for(int i=0;i<N;i++){
                for(int j=0;j<Specs[i];j++)IN[j]=OU[j];
                Layers[i]->EvaluateDer(IN,OU);
            }

            for(int j=0;j<Specs[N];j++)OUTPUT[j]=OU[j];
        }

        void SaveToFile(std::string FILENAME){
            std::ofstream file;
            file.open(FILENAME);
            std::cout.precision(16);
            file<<N<<"\n";
            file<<Specs[0]<<"\n";
            for(int i=1;i<N+1;i++){file<<Specs[i]<<"\n";}
            for(int i=0;i<N;i++){
                for(int j=0;j<Layers[i]->GetN();j++){
                    for(int k=0;k<Layers[i]->GetM();k++){
                        file<<Layers[i]->GetWeight(j,k)<<"\n";
                    }
                }
            }
            file.close();
        }

        void Mutate(T SDEV = T(1)/T(100)){
            for(int Layer=0;Layer<N;Layer++){
                for(int Node=0;Node<Specs[Layer+1];Node++){
                    T W[Specs[Layer]];
                    T ReNorm = 1;
                    T Mut;
                    for(int Weight=0;Weight<Specs[Layer];Weight++){
                        Mut = SDEV*RANDOM(-1,1);
                        W[Weight]  = this->GetWeight(Layer,Node,Weight);
                        W[Weight] += Mut;
                        ReNorm    += Mut;
                    }
                    for(int Weight=0;Weight<Specs[Layer];Weight++)W[Weight]/=ReNorm;
                    this->SetWeights(Layer,Node,W);
                }
            }
        } 

        void RecomputeWeights(T* IND, T* OUD, T (*Measure)(T), T Lambda){

            int M = GetM();

            T OUN[Specs[N]];
            T OUS[Specs[N]];
            T Deltas[Specs[N]];

            Evaluate(IND,OUN);
            EvaluateDer(IND,OUS);
            for(int j=0;j<M;j++){
                Deltas[j] = Measure((OUN[j]-OUD[j])/OUS[j]);
            }

            std::cout<<"This Datapoint will be propagated trough "<<N<<" layers"<<std::endl;

            for(int Layer=N-1;Layer>=0;Layer--){
                
                Matrix Wmunu(Specs[Layer+1],Specs[Layer]);
                Vector DFmuNBar(Specs[Layer+1]);
                Vector DFnuNMn1(Specs[Layer]);
                
                for(int Node=0;Node<Specs[Layer+1];Node++){
                    for(int Weight=0;Weight<Specs[Layer];Weight++){
                        Wmunu.Set(Node,Weight,Layers[Layer]->GetWeight(Node,Weight));
                    }
                }

            }

        }

        int GetN() const {return Specs[0];}
        int GetM() const {return Specs[N];}

    private:

        void DeepEvaluate(T* In, T* Out){
            // This procedure evaluates and stores the entire 
            // network, with all intermediate outputs 

        }
};

typedef NodeT<double> Node;
typedef LayerT<double> Layer;
typedef NetworkT<double> Network;

Network LoadFromFile(std::string FILE){ 
    std::ifstream file;
    std::string line;
    file.open(FILE,std::ifstream::in);
    getline(file,line);
    int NLayers = std::stoi(line);
    int NSpecs[NLayers+1];
    for(int i=0;i<NLayers+1;i++){
        getline(file,line);
        NSpecs[i]=stoi(line);
    }

    Network NEW(NLayers,NSpecs);
    
    for(int Layer=0;Layer<NLayers;Layer++){
        for(int Node=0;Node<NSpecs[Layer+1];Node++){
            double W[NSpecs[Layer]];
            for(int Weight=0;Weight<NSpecs[Layer];Weight++){
                getline(file,line);
                W[Weight] = stod(line);
            }
            NEW.SetWeights(Layer,Node,W);
        }
    }
    file.close();
    return NEW;
}

bool isperfectsq(int i){
    int j=i;
    int odd = 1;
    while(j>0){
        j   -= odd;
        odd += 2;
    }
    return ( j==0 ? 1 : 0);
}

bool isperfectcu(int i){
    int j=i;
    int lvl1 = 0;
    int lvl2 = 1;
    while(j>0){
        j    -= lvl2;
        lvl1 += 1;
        lvl2 += 6*lvl1;
    }
    return ( j==0 ? 1 : 0);
}

bool isdivby5(int i){
    return ( i%5 == 0 ? 1 : 0 );
}

template<class T>
class DataSetT{

    int N,M;
    std::vector<std::pair<std::vector<T>,std::vector<T>>> Data;

    public:

        DataSetT(int n, int m){
            N = n;
            M = m;
        }

        ~DataSetT(){};

        void Push_Back(T* In, T* Ou){
            std::vector<T> InAux;
            std::vector<T> OuAux;
            for(int i=0;i<N;i++)InAux.push_back(In[i]);
            for(int j=0;j<M;j++)OuAux.push_back(Ou[j]);
            Data.push_back(std::make_pair(InAux,OuAux));
        }

        int GetSize(){return Data.size();}
        int GetN(){return N;}
        int GetM(){return M;}

        std::pair<std::vector<T>,std::vector<T>> GetDataPoint(int i){return Data.at(i);}

        void Show(bool deep = false){
            std::cout<<"The size of the data set is: "<<Data.size()<<std::endl;
            for(auto d : Data){
                std::cout<<"{";
                for(auto i : d.first){
                    std::cout<<i<<" ";
                }
                std::cout<<"} -> {";
                for(auto i: d.second){
                    std::cout<<i<<" ";
                }
                std::cout<<"}"<<std::endl;
            }
        }
};

template<class T>
class TrainerT{

    NetworkT<T>* Trainee;
    DataSetT<T>* Dataset;
    int N,M;

    static T Measure(T Delta){return std::abs(Delta);}

    public:

        TrainerT(NetworkT<T>& NetPtr, DataSetT<T>& DataPtr){
            Trainee = &NetPtr;
            Dataset = &DataPtr;
            if( Dataset->GetN()!=Trainee->GetN() || Dataset->GetM()!=Trainee->GetM()){
                std::cout<<"Error: The dataset and network have incompatible sizes"<<std::endl;
                abort();
            }
            N = Dataset->GetN();
            M = Dataset->GetM();
        }

        ~TrainerT(){};

        void Train(T Lambda = T(1)/T(2)){
            
            std::pair<std::vector<T>,std::vector<T>> DataPoint;
            T In[N];
            T Ou[M];

            for(int i=0;i<Dataset->GetSize();i++){
                
                // std::cout<<"Deltas for Datapoint :"<<i<<std::endl;
                DataPoint = Dataset->GetDataPoint(i);
                for(int i=0;i<N;i++)In[i]=DataPoint.first.at(i);
                for(int j=0;j<M;j++)Ou[j]=DataPoint.second.at(j);
                
                Trainee->RecomputeWeights(In,Ou,Measure,Lambda);

            }

        }

        void ComputeDeltas(int DATA, T* Deltas){
            
        }

        void GetData(int i, T* Data){

        }

        void Show(bool deep = false){
            Trainee->Show(deep);
            Dataset->Show(deep);
        }

        void ShowNetwork(){Trainee->Show(true);}
        void ShowDataSet(){Dataset->Show(true);}
};

typedef DataSetT<double> DataSet;
typedef TrainerT<double> Trainer;

int main(int argc, char** argv){

    // Take 1 Parameter and return 3 Characteristics:
    // Is Perfect Squared 
    // Is Perfect Cube
    // Is Divisible by 5

    // Deep-Neural-Network Setup 

    #define NOu 3
    #define NIn 2

    int Specs[6] = {NIn,5,10,12,5,NOu};
    Network DNN(5,Specs);
    DNN.Mutate(1);
    
    // Dataset Setup

    #define NDATA 100
    DataSet DS1(NIn,NOu);
    for(int i=1;i<NDATA;i++){
        double In[NIn] = {double(i)/NDATA,double(i*i)/(NDATA*NDATA)};
        double Ou[NOu] = {double(isdivby5(i)),double(isperfectsq(i)),double(isperfectcu(i))};
        DS1.Push_Back(In,Ou);
    }

    // Trainer setup

    Trainer TR1(DNN,DS1);

    TR1.Train();


    DNN.SaveToFile("Network_1");


    
    return 0;

}