%uniformModelClassBuild

snnObj = uniformModel();

snnObj.useGPU = 1;

%% set variables
snnObj.seconds2simulate = 0.5; %in seconds

snnObj.nNeurons = 1000;
cellTypes = {'Pyramidal','Basket cell'};
nNeuronTypes = 2;


snnObj.aMean = [0.01,0.1]; %[Py,Bsk,Mar]
snnObj.aVar = [0.0015,0.0015]; %[Py,Bsk,Mar]
snnObj.dMean = [8,2]; %[Py,Bsk,Mar]
snnObj.dVar = [0,0]; %[Py,Bsk,Mar]

snnObj.percCellTypes = [0.8,0.2];

%snnObj.connectivityRule = {'ER','RC';'ER','RC'};
snnObj.connectivityRule = {'ER','ER';'ER','ER'};
%an ER graph with p >= log(nNeurons)/nNeurons is almost always
%fully connected (P. Erd�s, A. R�nyi, Publ. Math. Debrecen 6 (1959) 290.)

snnObj.connectivityRuleParamaters = cell(nNeuronTypes);  %cell mat nCellTypes by nCellTypes with respective paramaters


snnObj.percConnectivityMat = ...
    [0.25,0.23;...
     0.27,0.27]; %Py Bsk  by Py Bsk 
 
 
snnObj.weightsShapeMat = ...
    [4,4;...
     10,10];	 %Py Bsk  by Py Bsk     shape paramater a for a gamma distribution mean a*b and variance a*b^2, converges to gaussian with a > 10
weightsMeanMat = ...
    [0.2/120,0.25/120;...
     -2.5/5,-2.5/5];	 %Py Bsk  by Py Bsk     shape paramater a for a gamma distribution mean a*b and variance a*b^2, converges to gaussian with a > 10

%fitting a gamma given mean and variance to be lika a gaussian
%want a>10;, but gennerally as large as possible
%b = mean/a; 
%a = (mean^2)/var; so (mean^2)/var >= 10
%a^2 = mean/std >= sqrt(10) = 3.16; which should happen if the gaussian isn't
%going to be truncated anyway
 
 snnObj.weightsScaleMat = weightsMeanMat./snnObj.weightsShapeMat;
     %Py Bsk  by Py Bsk       scale paramater b for a gamma distribution mean a*b and variance a*b^2

 

snnObj.synapseTausMeanMat = ...
    [0.1,0.1 ; ...
     0.3,0.3];%Py Bsk  by Py Bsk       

 snnObj.synapseTausVarMat = ...
    [0.005,0.005 ; ...
     0.005,0.005];%Py Bsk  by Py Bsk       
 
snnObj.axonDelaysMinMat = ...
    [3,3;...
     2,2];   %Py Bsk Mar by Py Bsk Mar      axonDelay will be drawn as a random uniform distribtuion using this as the minimum
snnObj.axonDelaysMaxMat = ...
    [6,6;...
     4,4];   %Py Bsk Mar by Py Bsk Mar      axonDelay will be drawn as a random uniform distribtuion using this as the maximum
 
 
snnObj. nInputs = snnObj.nNeurons;
 
snnObj.inputWeightsShape = snnObj.weightsShapeMat(1,1);
snnObj.inputWeightsScale = (0.1/120)/snnObj.inputWeightsShape;
snnObj.inputPercConnectivity = snnObj.percConnectivityMat(1,1);
snnObj.inputConnectivityRule = 'ER';

percActive = 0;
snnObj.nInputsActive = floor(snnObj.nNeurons*percActive);
snnObj.inputTime_ms = 20; %ms

snnObj.verbose = 2;

snnObj.patchIndx = 1:100;

%{
    
    
 %}
 

%% built variables

%set a,d, nPy,nBsk,nMar
snnObj = buildNeurons(snnObj);

%set paramMatGPU, nConnections,
%synapseWeight_gpu, nrnPreIndexArray_gpu, nrnPostIndexArray_gpu, 
%postSynapseList_gpu, axonDelay_gpu, tau_gpu
snnObj = buildConnectivity(snnObj);

% set input paramaters 
% inputs, weightMatPre
snnObj = snnObj.buildInputConnectivity;

% set input firing times 
snnObj = snnObj.setInputs;

snnObj = snnObj.runSimulationGPU;

%{
%set v_gpu, u_gpu, spikeNeuron_gpu, axonState_gpu, synapseState_gpu, 
%dt_gpu (patches)
snnObj = initalizeSimululation(snnObj);
%}

%set spikes patches from the simualationsnnObj = runSimulation(snnObj)
        
%snnObj = analyszeSimululation(snnObj);