classdef  uniformModel < itzSimulation
    %the simplest model
    %neurons are hooked up using a simple connectivity rule with no other
    %considerations
    
    properties
        
        %cell mat nCellTypes by nCellTypes, though only consider the upper diagonal (symetry is assumed) erdosh reynie 'ER', reciprocal connectivity 'RC', small world 'SW', negi-nomial 'NN'
        connectivityRule
        
        %cell mat nCellTypes by nCellTypes with respective paramaters
        connectivityRuleParamaters
        
        
        %{
        aPyMean;
        aPyVar;
        aBskMean;
        aBskVar;
        aMarMean;
        aMarVar;
        dPyMean;
        dPyVar;
        dBskMean;
        dBskVar;
        dMarMean;
        dMarVar;
        %}
        
        %{
        percPy = 0.8;
        percBsk = 0.2;
        percMar = 0;
        
        nPy;
        nBsk;
        nMar;
        %}
        
        
        %nNeuronTypes by nNeuronTypes    shape paramater a for a gamma distribution mean a*b and variance a*b^2, gaussian with a = 10
        weightsShapeMat;
        
        %nNeuronTypes by nNeuronTypes      scale paramater b for a gamma distribution mean a*b and variance a*b^2
        weightsScaleMat;
        
        %nNeuronTypes by nNeuronTypes   synapseTau will be drawn as a gaussian using this as the mean
        synapseTausMeanMat;
        
        %nNeuronTypes by nNeuronTypes      synapseTau will be drawn as a gaussian using this as the variance
        synapseTausVarMat;
        
        %nNeuronTypes by nNeuronTypes      axonDelay will be drawn as a random uniform distribtuion using this as the minimum
        axonDelaysMinMat;
        
        %nNeuronTypes by nNeuronTypes      axonDelay will be drawn as a random uniform distribtuion using this as the maximum
        axonDelaysMaxMat;
        
        %{
        weightsPyPyMean
        weightsPyPyVar
        weightsPyBskMean
        weightsPyBskVar
        weightsPyMarMean
        weightsPyMarVar
        
        weightsBskPyMean
        weightsBskPyVar
        weightsBskBskMean
        weightsBskBskVar
        weightsBskMarMean
        weightsBskMarVar
        
        weightsMarPyMean
        weightsMarPyVar
        weightsMarBskMean
        weightsMarBskVar
        weightsMarMarMean
        weightsMarMarVar
        %}
        
        percConnectivityMat; %nNeuronTypes by nNeuronTypes
        %an ER graph with p >= log(nNeurons)/nNeurons is almost always
        %fully connected (P. Erd�s, A. R�nyi, Publ. Math. Debrecen 6 (1959) 290.)
        
        
        %extra input paramaters
        nInputsActive;      %number of the presynaptic neurons active per simulation when inputMode = 'Thal'
        inputIndx;             %1xnInputsActive indexs of the avtive input neurons
        inputWeightsShape
        inputWeightsScale
        inputPercConnectivity
        inputConnectivityRule = 'ER';
        %{
        percConnectPyPy
        percConnectPyBsk
        percConnectPyMar
        percConnectBskPy
        percConnectBskBsk
        percConnectBskMar
        percConnectMarPy
        percConnectMarBsk
        percConnectMarMar
        %}
        
    end
    
    properties (SetAccess = private)
        weightsMeanMat; %nNeuronTypes by nNeuronTypes
    end
    
    methods
        
        function uniformModelObj = uniformModel()
        end
        
        function uniformModelObj = buildConnectivity(uniformModelObj)
            %must have run build neurons before running this (or it will be done automatically)
            %sets paramMatGPU, nConnections,
            %synapseWeight_gpu, nrnPreIndexArray_gpu, nrnPostIndexArray_gpu,
            %postSynapseList_gpu, axonDelay_gpu, tau_gpu
            
            
            if isempty(uniformModelObj.nNeuronsByType)
                if uniformModelObj.verbose > 0
                    fprintf('uniformModelObj.nNeuronsByType is empty so rebuilding the neurons')
                end
                uniformModelObj = uniformModelObj.buildNeurons;
                if isempty(uniformModelObj.nNeuronsByType)
                    error('')
                end
            end
            
            %% initalize matrix for storring data with an upper estimate
            
            %this will help with the way matlab stores large data, but requires
            %some book keeping. we'll have to cut back this array after
            %we've built the matrix.
            
            %find an upper bound on the size of the matrix by considering
            %the mean and variance of the number of connections which
            %should follow a binomial when ER
            
            %binomeal mean n*p
            nConnectionsMean = uniformModelObj.nNeuronsByType*uniformModelObj.percConnectivityMat*uniformModelObj.nNeuronsByType';
            
            %binomeal variance np-np^2
            nConnectionsVar = nConnectionsMean - uniformModelObj.nNeuronsByType*(uniformModelObj.percConnectivityMat.^2)*uniformModelObj.nNeuronsByType';
            
            %estimate max size as 6 standard deviations above the mean
            maxConnectionEst = nConnectionsMean+6*sqrt(nConnectionsVar);
            
            
            if uniformModelObj.useGPU
                %over allocate array
                uniformModelObj.paramMatGPU = zeros(5,ceil(maxConnectionEst));
            else
                uniformModelObj.weightMat = sparse([],[],[],uniformModelObj.nNeurons,uniformModelObj.nNeurons,ceil(maxConnectionEst));
                uniformModelObj.synapticTau= sparse([],[],[],uniformModelObj.nNeurons,uniformModelObj.nNeurons,ceil(maxConnectionEst));
                uniformModelObj.axonDelayMat= sparse([],[],[],uniformModelObj.nNeurons,uniformModelObj.nNeurons,ceil(maxConnectionEst));
            end
            
            %% loop through connections within and between nNeuronTypes
            
            %used to offset indecies of pre and post synapic neurons
            %considering the overal network, not just a subnetwork
            %this is clearer in the code below
            indexShift = cumsum([0,uniformModelObj.nNeuronsByType(1:end-1)]);
            
            if uniformModelObj.useGPU
                %used to offset new connections in the global GPU array
                connectionIndx = 0;
            end
            
            if strcmp(uniformModelObj.connectivityRule, 'ER')
                
                for iPreNeuronType = 1:uniformModelObj.nNeuronTypes
                    for jPostNeuronType = 1:uniformModelObj.nNeuronTypes
                        
                        %% set connections indexed between these neuron types
                        if jPostNeuronType == iPreNeuronType
                            Pre2PostWeights = uniformModelObj.buildERconnectivity(...
                                uniformModelObj.nNeuronsByType(iPreNeuronType),...
                                uniformModelObj.nNeuronsByType(jPostNeuronType),...
                                uniformModelObj.percConnectivityMat(iPreNeuronType,jPostNeuronType));
                        else
                            Pre2PostWeights = uniformModelObj.buildERconnectivity(...
                                uniformModelObj.nNeuronsByType(iPreNeuronType),...
                                uniformModelObj.nNeuronsByType(jPostNeuronType),...
                                uniformModelObj.percConnectivityMat(iPreNeuronType,jPostNeuronType),'autaptic');
                        end
                        
                        
                        % change indexing to reflect the network
                        % ie. row 1 corresponds to pool i neurons
                        Pre2PostWeights(1,:) = Pre2PostWeights(1,:)+indexShift(iPreNeuronType);
                        Pre2PostWeights(2,:) = Pre2PostWeights(2,:)+indexShift(jPostNeuronType);
                        
                        
                        %% set connection paramaters
                        
                        %establish synaptic weights from a gamma distribution
                        connectionWeights = gamrnd(uniformModelObj.weightsShapeMat(iPreNeuronType,jPostNeuronType),uniformModelObj.weightsScaleMat(iPreNeuronType,jPostNeuronType),1,size(Pre2PostWeights,2));
                        
                        %taus from a normal distribution
                        taus = uniformModelObj.synapseTausMeanMat(iPreNeuronType,jPostNeuronType)+uniformModelObj.synapseTausVarMat(iPreNeuronType,jPostNeuronType)*randn(1,length(connectionWeights));
                        
                        %and delays from a uniform distribution
                        delays = randi([uniformModelObj.axonDelaysMinMat(iPreNeuronType,jPostNeuronType),uniformModelObj.axonDelaysMaxMat(iPreNeuronType,jPostNeuronType)],1,length(connectionWeights));
                        
                        
                        %% store connectivity into the struct for CPU or GPU simualations
                        if uniformModelObj.useGPU
                            
                            %presynaptic and post synaptic neurons
                            uniformModelObj.paramMatGPU(1:2,connectionIndx+(1:length(connectionWeights))) = Pre2PostWeights;
                            %synaptic weights
                            uniformModelObj.paramMatGPU(3,connectionIndx+(1:length(connectionWeights))) = connectionWeights;
                            %axon delays
                            uniformModelObj.paramMatGPU(4,connectionIndx+(1:length(connectionWeights))) = taus;
                            %synaptic taus
                            uniformModelObj.paramMatGPU(5,connectionIndx+(1:length(connectionWeights))) = delays;
                            
                            connectionIndx = connectionIndx+length(connectionWeights);
                        else
                            
                            uniformModelObj.weightMat(sub2ind(size(uniformModelObj.weightMat),Pre2PostWeights(1,:),Pre2PostWeights(2,:))) = connectionWeights;
                            uniformModelObj.synapticTau(sub2ind(size(uniformModelObj.weightMat),Pre2PostWeights(1,:),Pre2PostWeights(2,:))) = taus;
                            uniformModelObj.axonDelayMat(sub2ind(size(uniformModelObj.weightMat),Pre2PostWeights(1,:),Pre2PostWeights(2,:))) = delays;
                            
                        end
                        
                        
                    end
                end
                
            elseif strcmp(uniformModelObj.connectivityRule, 'ER_reciprocal')
                
                error('havent defined this connectivity rule yet')
                %do i to j and j to i in one step
                
            else
                error('havent defined this connectivity rule yet')
            end
            
            %% cut back the over allocated arrays
            if uniformModelObj.useGPU
                uniformModelObj.paramMatGPU(:,(connectionIndx+1):end) = [];
            end
            
        end
        
        function uniformModelObj = buildInputConnectivity(uniformModelObj)
            if strcmp(uniformModelObj.inputConnectivityRule,'ER')
                Pre2PostWeights = uniformModelObj.buildERconnectivity(uniformModelObj.nInputs,uniformModelObj.nNeurons,uniformModelObj.inputPercConnectivity,'autaptic');
                connectionWeights = gamrnd(...
                    uniformModelObj.inputWeightsShape,...
                    uniformModelObj.inputWeightsScale,...
                    1,size(Pre2PostWeights,2));
                uniformModelObj.inputWeightMat = sparse(Pre2PostWeights(1,:),Pre2PostWeights(2,:),connectionWeights,uniformModelObj.nInputs,uniformModelObj.nNeurons);
            else
                error('uniformModelObj.inputConnectivityRule must be set to ER for now')
            end
        end
        
        function uniformModelObj = setInputs(uniformModelObj,varargin)
            %by default selects a random subet of nInputsActive input neurons to fire
            %once within a set time inputTime
            %
            %can also fix the neuron identities with extra inputs
            %uniformModelObj.setInputs('fix')
            
            if strcmp(uniformModelObj.inputMode,'Thal')      
                
                if ~any(strcmp(varargin,'fix'))
                    try
                        uniformModelObj.inputIndx = sort(randperm(uniformModelObj.nInputs,uniformModelObj.nInputsActive));
                    catch
                        %need matlab 2013a or later
                        dummy = randperm(uniformModelObj.nInputs);
                        uniformModelObj.inputIndx = sort(dummy(1:uniformModelObj.nInputsActive));
                    end
                elseif uniformModelObj.verbose > 1
                    fprintf('using the inputIndx already in existance \n')
                end
                
                uniformModelObj.inputSpikes = cell(1,uniformModelObj.nInputs);
                uniformModelObj.inputSpikes(uniformModelObj.inputIndx) = cellfun(@(x) {randi(uniformModelObj.inputTime_ms)}, uniformModelObj.inputSpikes(uniformModelObj.inputIndx))
            else
                error('uniformModelObj.inputMode must be = to Thal')
            end
        end
        
        %% auxilary functions
        
        function Pre2PostWeights = buildERconnectivity(uniformModelObj,numPre,numPost,connectionProb,varargin)
            %outputs
            %Pre2PostWeight:    an nConnectionsx2 array of integers listing
            %                            the presynaptic indicies for row 1
            %                            and the postsynaptic indicies for
            %                            row2
            
            %Pre2PostWeights = buildSWconnectivity(numPre,numPost,connectionProb,1);
            
            if any(strcmp(varargin,'autaptic'))
                autaptic = 1;
            else
                if numPre == numPost
                    autaptic = 0;
                else
                    autaptic = 1;
                end
            end
            
            if autaptic
                nConnectionsPossible = numPre*numPost;
            else
                nConnectionsPossible = numPre*numPost - numPre;
            end
            
            %pull out indicies for connections which can be translated to
            %pre or post synaptic neuron IDs
            connectionIndicies = randi(nConnectionsPossible,1,nConnectionsPossible*connectionProb);
            
            %disallow autapses
            if ~autaptic
                connectionIndicies = connectionIndicies + floor(connectionIndicies/numPre);
            end
            
            %translate index fro the pre or post synaptic neuron indicies
            [preIndx,postIndx] = ind2sub([numPre,numPost],connectionIndicies);
            
            Pre2PostWeights = [preIndx;postIndx];
            
            
        end
        
        function Pre2PostWeights = buildERreciprocalconnectivity(uniformModelObj,numPre,numPost,connectionProb1,connectionProb2,recipProb)
            %{
            if any(strcmp(varargin,'autaptic'))
                autaptic = 1;
            else
                if numPre == numPost
                    autaptic = 0;
                else
                    autaptic = 1;
                end
             end
            
             if autaptic
                 nConnectionsPossible = numPre*numPost;
             else
                 nConnectionsPossible = numPre*numPost - numPre;
             end
             
             %pull out indicies for connections which can be translated to
             %pre or post synaptic neuron IDs
             connectionIndicies = randi(nConnectionsPossible,1,nConnectionsPossible*connectionProb1+nConnectionsPossible*connectionProb2);
             
             
             %disallow autapses
             if ~autaptic
                 connectionIndicies = connectionIndicies + floor(connectionIndicies/numPre);
             end
             
             
             
            %build unidirectional connections
            Pre2PostWeights = buildERconnectivity(numPre,numPost,connectionProb,varargin);
            
            %prune a subset of reciprocal connections to establish
            %unidirectional
           
            recip = [Pre2PostWeights,flipud(Pre2PostWeights)];
            %}
        end
        
        function Pre2PostWeights = buildSWconnectivity(uniformModelObj,numPre,numPost,connectionProb,rewireProb,varargin)
            % generalizes the notion of small world to feed forward networks
            % uses the idea of connection probability opposed to nearest neighbors
            %
            % NOTE: this algorithm preserves out degree from nodes
            % NOTE: this should be modified to deal with sparce matrices
            % for large number of neurons
            
            if any(strcmp(varargin,'autaptic'))
                autaptic = 1;
            else
                if numPre == numPost
                    autaptic = 0;
                else
                    autaptic = 1;
                end
            end
            
            % map the connection probability to the number of nearest neighbors
            numPre2PostNeighbors = ceil(connectionProb*numPost);
            
            Pre2PostWeights = zeros(numPre,numPost);
            
            %a template connection row to be shifted
            if autaptic
                Pre2PostConnectionRow = [ones(1,numPre2PostNeighbors),zeros(1,numPost-numPre2PostNeighbors)];
            else
                Pre2PostConnectionRow = [ones(1,floor(numPre2PostNeighbors/2)),0,ones(1,ceil(numPre2PostNeighbors/2)),zeros(1,numPost-numPre2PostNeighbors-1)];
            end
            
            shift = -floor(numPre2PostNeighbors/2); %a baseline shift
            Post2PreRatio = (numPost/numPre); %for finding how much to shift for each pre neuron
            
            for iexcit = 1:numPre
                
                %rewire each edge with probability rewireProb
                this_Pre2PostConnectionRow = Pre2PostConnectionRow;
                rewires = find(rand(1,numPre2PostNeighbors)<=rewireProb);
                if isempty(rewires)
                    
                else
                    open2unwire = find(this_Pre2PostConnectionRow == 1);
                    this_Pre2PostConnectionRow(open2unwire(rewires)) = 0;
                    
                    open2wire = find(this_Pre2PostConnectionRow == 0);
                    newWireOrder = open2wire(randperm(length(open2wire)));
                    this_Pre2PostConnectionRow(newWireOrder(1:length(rewires))) = 1;
                end
                
                Pre2PostWeights(iexcit,1:numPost) = circshift(this_Pre2PostConnectionRow',  round(shift))';
                
                if ~autaptic %rewire if autapse
                    if Pre2PostWeights(iexcit,iexcit) == 1
                        
                        open2wire = find(Pre2PostWeights(iexcit,1:numPost) == 0);
                        newWireIndx = open2wire(ceil(rand(1,1)*length(open2wire)));
                        
                        Pre2PostWeights(iexcit,newWireIndx) = 1;
                        Pre2PostWeights(iexcit,iexcit) = 0;
                        
                    end
                end
                
                %rotate template each next row
                
                if Post2PreRatio > 1 %then incrementing by several each time
                    if round(mod(iexcit,2))
                        shift = shift+ceil(Post2PreRatio);
                    else
                        shift = shift+floor(Post2PreRatio);
                    end
                else %then some repeats taken care of by the round(shift)
                    shift = shift+Post2PreRatio;
                end
                
            end
            
            
        end
        
    end
    
end
