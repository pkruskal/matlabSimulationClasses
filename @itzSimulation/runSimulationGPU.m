function itzSimulationObj = runSimulationGPU(itzSimulationObj)

testStates = 0;

%% initalize CPU variables
timeArray = 1:itzSimulationObj.dt:(1000*itzSimulationObj.seconds2simulate);
synapseMatPre = zeros(itzSimulationObj.nInputs,itzSimulationObj.nNeurons);
itzSimulationObj.patch = zeros(length(itzSimulationObj.patchIndx),length(timeArray));
itzSimulationObj.spikes = cell(1,itzSimulationObj.nNeurons);
%% initalize kernals

itzSimulationObj.nConnections = size(itzSimulationObj.paramMatGPU,2);

threadsPerBlock = 128;
nBlocks = ceil(itzSimulationObj.nConnections/threadsPerBlock);
nThreads = threadsPerBlock*nBlocks;
connectionsPerThread = ceil(itzSimulationObj.nConnections/(nThreads*nBlocks));

nConnections_gpu = gpuArray(int32(itzSimulationObj.nConnections));
connectionsPerThread_gpu = gpuArray(int32(connectionsPerThread));
updateNeuronState = parallel.gpu.CUDAKernel('itzSim20130406.ptx', 'itzSim20130406.cu','updateNeuronState');
updateSynapticState = parallel.gpu.CUDAKernel('itzSim20130406.ptx', 'itzSim20130406.cu','updateSynapticState');
updateNeuronState.GridSize = itzSimulationObj.nNeurons;
updateSynapticState.ThreadBlockSize = threadsPerBlock;
updateSynapticState.GridSize = nBlocks;
%% set GPU variables

%{
 5x nConnection matrix
        row 1 a list of presynaptic neurons
        row 2 a post synaptic neurons
        row 3 synaptic weights
        row 4 axonDelay
        row 5 synapticTau

        histc on row 1 will give nrnPreIndexArray
        histc on row 2 will give all nrnPosIndexArray?

        sort the list of presnaptics,
        take the list of postSynaptics as postSynapseList

        sort the list of postsnaptics,
        take rows 3,4,5 as the weights, delay and tau arrays
%}

%histc to count the number of times this is a post synaptic neuron, to
%obtain the number of presynaptic connections to account for
nrnPreIndexArray = histc(itzSimulationObj.paramMatGPU(2,:),0:1:itzSimulationObj.nNeurons);
itzSimulationObj.nrnPreIndexArray_gpu = gpuArray(int32(nrnPreIndexArray-1));
clear nrnPreIndexArray

%histc to count the number of times this is a pre synaptic neuron, to
%obtain the number of postsynaptic connections to account for
nrnPostIndexArray = histc(itzSimulationObj.paramMatGPU(1,:),0:1:itzSimulationObj.nNeurons);
itzSimulationObj.nrnPostIndexArray_gpu = gpuArray(int32(nrnPostIndexArray-1));
clear nrnPostIndexArray

%synaps state arrays are ordered by the incoming connections to the post
%synaptic neuorns, so sort by the post synaptic neuron
[itzSimulationObj.paramMatGPU(2,:),postSortIndx] = sort(itzSimulationObj.paramMatGPU(2,:));
itzSimulationObj.paramMatGPU(1,postSortIndx);
itzSimulationObj.paramMatGPU(3,:) = itzSimulationObj.paramMatGPU(3,postSortIndx);
itzSimulationObj.paramMatGPU(4,:) = itzSimulationObj.paramMatGPU(4,postSortIndx);
itzSimulationObj.paramMatGPU(4,:) = itzSimulationObj.paramMatGPU(5,postSortIndx);

%to index the synapse array given a presynaptic neuron, take the post
%sorted list, and the indexes using find
%% NEED TO CHECK THAT THIS WORKS
[~,synapseArrayIndx] = sort(itzSimulationObj.paramMatGPU(1,:));
itzSimulationObj.postSynapseList_gpu = gpuArray(int32(synapseArrayIndx-1));

itzSimulationObj.axonDelay_gpu = gpuArray(int32(itzSimulationObj.paramMatGPU(4,:)));
itzSimulationObj.synapseWeight_gpu = gpuArray(single(itzSimulationObj.paramMatGPU(2,:)));
itzSimulationObj.tau_gpu = gpuArray(single(itzSimulationObj.paramMatGPU(5,:)));


itzSimulationObj.axonState_gpu = gpuArray(zeros(1,itzSimulationObj.nConnections,'int32'));
itzSimulationObj.synapseState_gpu = gpuArray(zeros(1,itzSimulationObj.nConnections,'single'));
itzSimulationObj.v_gpu = gpuArray(single(ones(1,itzSimulationObj.nNeurons)*-70));
itzSimulationObj.u_gpu = gpuArray(zeros(1,itzSimulationObj.nNeurons,'single'));

%itzSimulationObj.a_gpu = gpuArray(single(itzSimulationObj.a));
%itzSimulationObj.d_gpu = gpuArray(single(itzSimulationObj.d));
itzSimulationObj.I_gpu = gpuArray(single(zeros(1,itzSimulationObj.nNeurons)));
itzSimulationObj.dt_gpu =  gpuArray(single(itzSimulationObj.dt));


spikeNeuron_gpu = gpuArray(int32(zeros(1,itzSimulationObj.nNeurons)));
%% simulation
tic

for iTime = timeArray    

    fprintf([ 'time ' num2str(iTime) ' took ' num2str(toc) '\n'])
    
    %% update inputs
    
        %{
    %update inputs
    if strcmp(inputMode,'set')
        if iTime <= itzSimulationObj.preTime
            activeNeurons = find(inputMat(:,iTime) == 1);
            itzSimulationObj.synapseMatPre(activeNeurons,:) = itzSimulationObj.synapseMatPre(activeNeurons,:)+itzSimulationObj.weightMatPre(activeNeurons,:);
        end
        itzSimulationObj.synapseMatPre = itzSimulationObj.synapseMatPre-itzSimulationObj.synapseMatPre.*itzSimulationObj.synapticTau;
    end
    
    % set new inputs to be done on the CPU/GPU
    if iTime <= itzSimulationObj.preTime
        inputs(1:excitSize) = sum(itzSimulationObj.synapseMatPre)';
        
        if useGPU
             itzSimulationObj.I_gpu = gpuArray(single(inputs));
        end
    elseif any(inputs>0)
        inputs(1:excitSize) = zeros(1,excitSize);
         itzSimulationObj.I_gpu = gpuArray(single(inputs));
    end
    %}

    if strcmp(itzSimulationObj.inputMode,'Thal')
        if iTime <= itzSimulationObj.inputTime_ms
            
            %{
            %array of 1's and 0's for this time step indicating if a neuron fired
            %or not using the inputSpikes cell array
            inputBoolArray = cell2mat(cellfun(@(x) ~isempty(x) & x >= iTime & x < iTime+itzSimulationObj.dt,itzSimulationObj.inputSpikes,'uniformoutput',false));
            
            %update the synapse mat for the inputs
            synapseMatPre(inputBoolArray,:) = inputWeightMat(inputBoolArray,:);
            %}
            
            for iInput = 1:itzSimulationObj.nInputs
                if ~isempty(itzSimulationObj.inputSpikes{iInput})
                    if any(itzSimulationObj.inputSpikes{iInput} >= iTime && itzSimulationObj.inputSpikes{iInput} < iTime+itzSimulationObj.dt)
                        synapseMatPre(iInput,:) = itzSimulationObj.inputWeightMat(iInput,:);
                    end
                end
            end
            
            
        end
        
        if iTime <= itzSimulationObj.inputTime_ms + 50
            synapseMatPre = synapseMatPre-synapseMatPre.*itzSimulationObj.synapseTausMeanMat(1);
            itzSimulationObj.I_gpu = gpuArray(single(sum(synapseMatPre)'));
        elseif iTime < itzSimulationObj.inputTime_ms + 55
            %make sure everything is set to 0
            itzSimulationObj.I_gpu = gpuArray(zeros(1,size(synapseMatPre,2),'single'));
        end
        
    end
    
    %% simulate neurons
    time_gpu = gpuArray(single(iTime));
    
    %call GPU
    [spikeNeuron_gpu,...
        itzSimulationObj.nrnPreIndexArray_gpu, ....
        itzSimulationObj.nrnPostIndexArray_gpu,...
        itzSimulationObj.postSynapseList_gpu, ...
        itzSimulationObj.axonState_gpu,...
        itzSimulationObj.axonDelay_gpu,...
        itzSimulationObj.synapseState_gpu, ...
        itzSimulationObj.I_gpu, ...
        itzSimulationObj.v_gpu, ...
        itzSimulationObj.u_gpu, ...
        itzSimulationObj.a_gpu, ...
        itzSimulationObj.d_gpu,...
        itzSimulationObj.dt_gpu, ...
        time_gpu] ...
        = feval(updateNeuronState,...
        spikeNeuron_gpu,...
        itzSimulationObj.nrnPreIndexArray_gpu, ....
        itzSimulationObj.nrnPostIndexArray_gpu,...
        itzSimulationObj.postSynapseList_gpu, ...
        itzSimulationObj.axonState_gpu,...
        itzSimulationObj.axonDelay_gpu,...
        itzSimulationObj.synapseState_gpu, ...
        itzSimulationObj.I_gpu, ...
        itzSimulationObj.v_gpu, ...
        itzSimulationObj.u_gpu, ...
        itzSimulationObj.a_gpu, ...
        itzSimulationObj.d_gpu,...
        itzSimulationObj.dt_gpu, ...
        time_gpu);
    
    %% store the results
    
    %add spikes
    spikeNeuron = gather(spikeNeuron_gpu);
    itzSimulationObj.spikes(find(spikeNeuron == 1)) = arrayfun(@(x) {[x{:},((iTime-1)*itzSimulationObj.dt)/1000]},itzSimulationObj.spikes(find(spikeNeuron == 1)));
    
    %add voltage clamps
    v = gather(itzSimulationObj.v_gpu);
    v(find(spikeNeuron == 1)) = 40;
    itzSimulationObj.patch(:,iTime) = v(itzSimulationObj.patchIndx);
    
    if  testStates == 1
        theseAxonStates = gather(itzSimulationObj.axonState_gpu);
        theseSynapseStates = gather(itzSimulationObj.synapseState_gpu);
        
        testIndx = 1;
        for ineuron = 1:length(testNeurons)
            for iConnection = nrnPostIndexArray(ineuron):1:(nrnPostIndexArray(ineuron+1)-1)
                testAxons3(testIndx,iTime) = theseAxonStates(postSynapseList(iConnection));
                testSynapse3(testIndx,iTime) = theseSynapseStates(postSynapseList(iConnection));
                testIndx = testIndx+1;
            end
        end
        
    end
    
    %% simulate connections
    
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
    
    [connectionsPerThread_gpu,...
        nConnections_gpu,...
        itzSimulationObj.synapseState_gpu,...
        itzSimulationObj.tau_gpu,...
        itzSimulationObj.axonState_gpu,...
        itzSimulationObj.synapseWeight_gpu] = ...
        feval(updateSynapticState ,...
        connectionsPerThread_gpu,...
        nConnections_gpu,...
        itzSimulationObj.synapseState_gpu,...
        itzSimulationObj.tau_gpu,...
        itzSimulationObj.axonState_gpu,...
        itzSimulationObj.synapseWeight_gpu);
    
    %{
    %check to see if break
    if iTime > (100+itzSimulationObj.preTime) && sum(sum(axonState>0)) == 0
        if quiesentTime >= 300; %need to allow for inhibitory rebound effects?
            fprintf('quiesense terminated simulation \n')
            break
        else
            quiesentTime = quiesentTime+1;
        end
    else
        quiesentTime = 0; %how long quiesense is needed for the program to terminate
    end
    %}
    
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

%reset(gpuVar)

