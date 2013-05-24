seconds2simulate
dt

inputMode
preTime
inputs
synapseMatPre





for iTime = 1:dt:(1000*seconds2simulate)
    
    tic
   
    %update inputs
    if strcmp(inputMode,'set')
        if iTime <= simulation.preTime
            activeNeurons = find(inputMat(:,iTime) == 1);
            simulation.synapseMatPre(activeNeurons,:) = simulation.synapseMatPre(activeNeurons,:)+simulation.weightMatPre(activeNeurons,:);
        end
        simulation.synapseMatPre = simulation.synapseMatPre-simulation.synapseMatPre.*simulation.synapticTau;
    end
    
    % set new inputs to be done on the CPU/GPU
    if iTime <= simulation.preTime
        inputs(1:excitSize) = sum(simulation.synapseMatPre)';
        
        if useGPU
             I_gpu = gpuArray(single(inputs));
        end
    elseif any(inputs>0) 
        inputs(1:excitSize) = zeros(1,excitSize);
         I_gpu = gpuArray(single(inputs));
    end
        
    
    %% simulate neurons
        time_gpu = gpuArray(single(iTime));
       
        %call GPU
        [spikeNeuron_gpu,...
            nrnPreIndexArray_gpu, ....
            nrnPostIndexArray_gpu,...
            postSynapseList_gpu, ...
            axonState_gpu,...
            axonDelay_gpu,...
            synapseState_gpu, ...
            I_gpu, ...
            v_gpu, ...
            u_gpu, ...
            a_gpu, ...
            d_gpu,...
            dt_gpu, ...
            time_gpu] ...
            = feval(updateNeuronState,...
            spikeNeuron_gpu,...
            nrnPreIndexArray_gpu, ....
            nrnPostIndexArray_gpu,...
            postSynapseList_gpu, ...
            axonState_gpu,...
            axonDelay_gpu,...
            synapseState_gpu, ...
            I_gpu, ...
            v_gpu, ...
            u_gpu, ...
            a_gpu, ...
            d_gpu,...
            dt_gpu, ...
            time_gpu);
        
        
        %add spikes
        spikeNeuron = gather(spikeNeuron_gpu);
        spikes(find(spikeNeuron == 1)) = arrayfun(@(x) {[x{:},((iTime-1)*simulation.dt)/1000]},spikes(find(spikeNeuron == 1)));

        %add voltage clamps
        v = gather(v_gpu);
        v(find(spikeNeuron == 1)) = 40;
        V(:,iTime) = v;
        
        if  testStates == 1
            theseAxonStates = gather(axonState_gpu);
            theseSynapseStates = gather(synapseState_gpu);
            
            testIndx = 1;
            for ineuron = 1:length(testNeurons)
                for iConnection = nrnPostIndexArray(ineuron):1:(nrnPostIndexArray(ineuron+1)-1)
                    testAxons3(testIndx,iTime) = theseAxonStates(postSynapseList(iConnection));
                    testSynapse3(testIndx,iTime) = theseSynapseStates(postSynapseList(iConnection));
                    testIndx = testIndx+1;
                end
            end
            
        end
        
    
    %{  
__global__ void updateNeuronState(int *nrnPreIndexArray, int *postSynapseList, int *nrnPostIndexArray, int *axonState, int *axonDelay, float *synapseState, float *I, float *v, float *u, float *a, float *d, float *dt, float *time) {
int *nrnPostIndexArray,	1xnNeurons+1 start indicies for the post synaptic neuron list
int *postSynapseList, 1xnConnections lists the post synaptic neuron targets for every neuron, indexed by nrnIndexArray
int *nrnPreIndexArray,	1xnNeurons+1 start indicies for the pre synaptic neuron input list
int *axonState, 1xnConnections ordered by the postsynaptic neurons
int *axonDelay, 1xnConnections ordered by the postsynaptic neurons
float *synapseState, 1xnConnections ordered by the postsynaptic neurons
float *I, 1xnNeurons
float *v, 1xnNeruons
float *u, 1xnNeurons
float *a, 1xnNeurons
float *d, 1xnNeurons
float *dt, 1x1
float *time 1x1
    %}  
    %{
//in CUDA
// Kernal 1 neural integrasion


    for (iConnection = nrnPreIndexArray(neuronIndex); iConnection < nrnPreIndexArray(neuronIndex+1); iConnection++){
        if (synapseState(iConnection) > 0)
            I(neuronIndex) = I(neuronIndex) + synapseState(iConnection)*(50-v(neuronIndex));
        else if (synapseState(iConnection) < 0)
            I(neuronIndex) = I(neuronIndex) - synapseState(iConnection)*(-75-v(neuronIndex));
    }
    
        

v(neuronIndex)=v(neuronIndex)+0.5*dt*((0.04*v(neuronIndex)+5)*v(neuronIndex)+140-u(neuronIndex)+I(neuronIndex);    // for numerical
v(neuronIndex)=v(neuronIndex)+0.5*dt*((0.04*v(neuronIndex)+5)*v(neuronIndex)+140-u(neuronIndex)+I(neuronIndex);    // stability time
u(neuronIndex)=u(neuronIndex)+a(neuronIndex)*(0.2*v(neuronIndex)-u(neuronIndex));                   // step is 0.5 ms

if (v(neuronIndex) > 40){
    v(neuronIndex) = -65;
    u(neuronIndex) = u(neuronIndex) + d(neuronIndex);
    spikeNeuron = neuronIndex;
    spikeTime = t;
    for (iConnection = nrnPostIndexArray(neuronIndex); iConnection < (nrnPostIndexArray(neuronIndex+1)-1);iConnection++){
        axonState(postSynapseList(iConnection)) = axonDelay(postSynapseList(iConnection));
    }
}
    %}
    
    %% simulate connections
    
    [connectionsPerThread_gpu,...
        nConnections_gpu,...
        synapseState_gpu,...
        tau_gpu,...
        axonState_gpu,...
        synapseWeight_gpu] = ...
        feval(updateSynapticState ,...
        connectionsPerThread_gpu,...
        nConnections_gpu,...
        synapseState_gpu,...
        tau_gpu,...
        axonState_gpu,...
        synapseWeight_gpu);
    
    fprintf([ 'time ' num2str(iTime) ' took ' num2str(toc) '\n'])
    
    %check to see if break
    if iTime > (100+simulation.preTime) && sum(sum(axonState>0)) == 0 
        if quiesentTime >= 300; %need to allow for inhibitory rebound effects?
            fprintf('quiesense terminated simulation \n')
            break
        else
            quiesentTime = quiesentTime+1;
        end
    else
        quiesentTime = 0; %how long quiesense is needed for the program to terminate
    end
    
    %{
float *synapseState 1xnConnections
float *synapseWeight 1xnConnections
float *tau 1xnConnections
    %}
    
    %{
// Kernal 2 connectivity updates
     connectionsPerThread = ceil(nConnections/(nThreads*nBlocks));
        
        for (iConnection = (iThread-1)*connectionsPerThread; iConnection < 9iThread*connectionsPerThread); iConnection++)
            
            if iConnection <= nConnections
                {
for (iConnection = nrnIndexArray(neuronIndex); iConnection < nrnIndexArray(neuronIndex+1);iConnection++){
    if (synapseState(iConnection) > 0)
    {
        synapseState(iConnection) = synapseState(iConnection)-synapseState(iConnection)*tau(iConnection);
    }
    else if (synapseState(iConnection) < 0)
    {
        synapseState = 0;
    }
    
    if (axonState(iConnection) > 0)
    {
            axonState(iConnection)--;
    }
        else if (axonState(iConnection) == 0)
    {
            synapseState(iConnection) = synapseState(iConnection) + synapseWeight(iConnection);
    }
        
        }
}

    %}
    
end

reset(gpuVar)
