function itzSimulationObj = runCPUSimulation(itzSimulationObj)


%% Define various Network Parameters

%variable storing the presynaptic weight matrix
inputWeightMat=sparse([],[],[],itzSimulationObj.nNeurons,itzSimulationObj.nNeurons);

%dynamic variable storing synaptic presynaptic states for the inputs
synapseMatPre = sparse([],[],[],size(itzSimulationObj.inputWeightMat,1),size(itzSimulationObj.inputWeightMat,2));

s_nzmax=sum(itzSimulationObj.weightMat(:)~=0);
%dynamic variable storing synaptic states
synapseMat = sparse([],[],[],size(itzSimulationObj.weightMat,1),size(itzSimulationObj.weightMat,2),s_nzmax);

% storage variable collecting spike times of the network
spikes = cell(1,itzSimulationObj.nNeurons);

%dynamic variable storing axon conductance delays
axonState = sparse([],[],[],size(itzSimulationObj.weightMat,1),size(itzSimulationObj.weightMat,2),sum(itzSimulationObj.weightMat(:)~=0));
axonState(itzSimulationObj.weightMat(:)~=0) = 0.5;


v = -70*ones(1,itzSimulationObj.nNeurons)'; %itz voltage paramater for neurons
u = zeros(1,itzSimulationObj.nNeurons)';       %itz conductance paramater for neurons
c = -65;                                            %itz voltage reset paramater for neurons
b = 0.2;                                            %itz conductance  dynamics paramater for neurons



%% some error checks (add to this as necessary)
if itzSimulationObj.nNeurons ~= size(weightMat,1)
    error('weightMat needs a row for each neuron')
elseif size(itzSimulationObj.weightMat,1) ~= size(itzSimulationObj.weightMat,2)
    error('weightMat must be a square matrix with dimention nNeurons')
elseif size(itzSimulationObj.weightMat) ~= size(itzSimulationObj.axonDelayMat)
    error('weightMat and delayMat must be the same size')
elseif length(itzSimulationObj.a) ~= itzSimulationObj.nNeurons
    error('length(a) must = nNeurons')
elseif length(itzSimulationObj.a) ~= length(itzSimulationObj.d)
    error('a and d must be the same size')
end

%% Begin Simulation

for time = 1:totalTime
    
    %% update input
    
    %array of 1's and 0's for this time step indicating if a neuron fired
    %or not using the inputSpikes cell array
    inputBoolArray = cell2mat(cellfun(@(x) x > time & x < time,inputSpikes,'uniformoutput',false));
    
    %update the synapse mat for the inputs
    synapseMatPre(inputBoolArray,:) = inputWeightMat(inputBoolArray,:);
    
    %% update conductances for the input to neurons
    gexcit = sum(synapseMat(1:excitSize,:))';
    ginhib = sum(synapseMat(excitSize+1:inhibSize,:))';
    gexcit(1:excitSize) = gexcit(1:excitSize) + sum(synapseMatPre(1:excitSize))';
    I = gexcit.*(50-v) - ginhib.*(-75-v);
    
    %% update neurons
    v=v+0.5*dt*((0.04*v+5).* v+140-u+I(:,end));         % for numerical
    v= v+0.5* dt*((0.04* v+5).* v+140- u+I(:,end));    % stability time
    u= u+ a.*(b* v- u);                                           % step is 0.5 ms
    
    %% check for firing and reset of neurons
    firedIndx = v > threshold;
    v(firedIndx) = c;
    u(firedIndx) = u(firedIndx) + d(firedIndx);
    spikes(firedIndx) = cellfun(@(x) [x,time], spikes(firedIndx),'uniformoutput',false);
    
    %% update synapses
    
    
    %input synapic timecourse
    synapseMatPre = synapseMatPre-synapseMatPre.*synapticTau;
    %excitatory to excitatory synapses timecourse
     synapseMat(1: excitSize,1: excitSize) =  synapseMat(1: excitSize,1: excitSize) ...
        - synapseMat(1: excitSize,1: excitSize).* synapticTau; %exponential decay
    %inhibitory to excitatory synapses timecourse
     synapseMat( excitSize+(1: inhibSize),1: excitSize) =  synapseMat( excitSize+(1: inhibSize),1: excitSize) ...
        - synapseMat( excitSize+(1: inhibSize),1: excitSize).* synapticTauI2E; %exponential decay
    %excitatory to inhibitory synapses timecourse
     synapseMat(1: excitSize, excitSize+(1: inhibSize)) =  synapseMat(1: excitSize, excitSize+(1: inhibSize)) ...
        - synapseMat(1: excitSize, excitSize+(1: inhibSize)).* synapticTauE2I; %exponential decay
    
    %axon conductance delay (axonState counts down to 1 to signal that the AP has reached the synapse)
    axonState( axonState>0) = axonState( axonState>0) - 1;
    
    %add APs to the synaptic state (axonState counts down to 1 to signal that the AP has reached the synapse)
    apIndx =  axonState > 0 &  axonState < 1; 
     synapseMat(apIndx) =  synapseMat(apIndx) ...
        + weightMat(apIndx);
    
    %send AP down the axon when the cell fired (ie. update axonState matrix to it's axon delay paramater)
     axonState(firedIndx,:) =  delayMat(firedIndx,:);
    
    
end

end