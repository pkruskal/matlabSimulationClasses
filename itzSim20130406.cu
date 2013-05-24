#include <stdio.h>
#include "cudabyexample/common/book.h"

//#define NUMNEURNS	10000
//#define TIME 1000

//const int neuronsPerBlock
//const int blocksPerGrid = imin( 32, (NUMNEURNS+neuronsPerBlock-1) / neuronsPerBlock );

//#define INPUT 0

//itzSim20130406


__global__ void test(float *I) {
	int neuronIndex;
	neuronIndex = threadIdx.x + blockIdx.x * blockDim.x; 
	
	I[neuronIndex] = 0;
}


__global__ void updateNeuronState(int *spikeNeuron, int *nrnPreIndexArray, int *nrnPostIndexArray, int *postSynapseList, int *axonState, int *axonDelay, float *synapseState, float *I, float *v, float *u, float *a, float *d, float *dt, float *time) {

/*
int *spikeNeuron, 1xnNeurons 1s or 0s if a spike fired or not (can do a bool?)
int *nrnPostIndexArray,	1xnNeurons+1 start indicies for the post synaptic neuron list
int *nrnPreIndexArray,	1xnNeurons+1 start indicies for the pre synaptic neuron list
int *postSynapseList, 1xnConnections lists the post synaptic neuron targets for every neuron, indexed by nrnIndexArray
int *axonState, 1xnConnections
int *axonDelay, 1xnConnections
float *synapseState, 1xnConnections
float *I, 1xnNeurons
float *v, 1xnNeruons
float *u, 1xnNeurons
float *a, 1xnNeurons
float *d, 1xnNeurons
float *dt, 1x1
float *time 1x1
*/

	int neuronIndex;
	
	neuronIndex = threadIdx.x + blockIdx.x * blockDim.x; 

	// add incoming currents
	float synapticInput = (50 - v[neuronIndex]) * I[neuronIndex];

	// update synaptic input
	for (int iConnection = nrnPreIndexArray[neuronIndex]; iConnection < nrnPreIndexArray[neuronIndex+1]; iConnection ++){
        if (synapseState[iConnection] > 0){
            synapticInput = synapticInput + synapseState[iConnection] * (50 - v[neuronIndex]);
			}
        else if (synapseState[iConnection] < 0){
            synapticInput = synapticInput - synapseState[iConnection] * (-75 - v[neuronIndex]);
			}
    }
    
	//synapticInput = 0.0;
	//v[neuronIndex] = synapticInput;


	v[neuronIndex] = v[neuronIndex] + 0.5 * dt[0] * ( (0.04 * v[neuronIndex] + 5 ) * v[neuronIndex] + 140 - u[neuronIndex] + synapticInput );    // for
	v[neuronIndex] = v[neuronIndex] + 0.5 * dt[0] * ( (0.04 * v[neuronIndex] + 5 ) * v[neuronIndex] + 140 - u[neuronIndex] + synapticInput );    // stability time
	u[neuronIndex] = u[neuronIndex] + a[neuronIndex] * (0.2 * v[neuronIndex]-u[neuronIndex]);                   // step is 0.5 ms
	      
	
	
	spikeNeuron[neuronIndex] = 0;
	if (v[neuronIndex] > 40.0){
		v[neuronIndex] = -65.0;
		u[neuronIndex] = u[neuronIndex] + d[neuronIndex];
		spikeNeuron[neuronIndex] = 1;
		for (int iConnection = nrnPostIndexArray[neuronIndex]; iConnection < nrnPostIndexArray[neuronIndex+1]; iConnection ++){
			axonState[postSynapseList[iConnection]] = axonDelay[postSynapseList[iConnection]];
		}

	
	}

	
		
}



__global__ void updateSynapticState(int *connectionsPerThread, int *nConnections, float *synapseState, float *tau, int *axonState, float *synapseWeight) {

/*
int   *axonState 1xnConnections
float *synapseState 1xnConnections
float *synapseWeight 1xnConnections
float *tau 1xnConnections
*/

	int connectionSet = threadIdx.x + blockIdx.x * blockDim.x; 

    
    //connectionsPerThread = ceil(nConnections/(gridDim.x * blockDim.x));
    
	int iConnection;
	
    for (iConnection = 
	(connectionSet * connectionsPerThread[0]);
	 iConnection < ((connectionSet+1) * connectionsPerThread[0]);
	  iConnection ++){
        
        if (iConnection < nConnections[0]){
            
            if (synapseState[iConnection] != 0) {
                synapseState[iConnection] = synapseState[iConnection]-synapseState[iConnection]*tau[iConnection];
				}
            else if (synapseState[iConnection] > -0.0001 && synapseState[iConnection] < -0.0001)
			{
                synapseState = 0;
            }
            
            if (axonState[iConnection] > 1){
                axonState[iConnection] = axonState[iConnection]-1;
				}
            else if (axonState[iConnection] == 1){
                axonState[iConnection] = axonState[iConnection]-1;
                synapseState[iConnection] = synapseState[iConnection] + synapseWeight[iConnection];
				}
            else if (axonState[iConnection] < 0){
                axonState[iConnection] = 0;
            }

        }

    }

}


