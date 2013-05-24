classdef itzSimulation
    % a general network simulation class for the itzikevich type neuron
    % specific types of simulation questions should inherit this class as a
    % starting point 
    %
    % - this class handels very general paramaters and the simulation
    % paramaters
    %
    % - this class handels the itzikevich neuron paramaters and their
    % construction, only itz paramaters a, and d as random variables for now, the rest are
    % fixed
    %
    % - this class also impliments the simulation on either the CPU or GPU
    %
    % - this class also supports different modes of input, more complicated
    % modes of input must over ride the simulation function in subclasses
    %
    % simulation subclasses are responsible for setting connectivity
    % variables, as well as synapse variables for the simulation 
    % see the gpu and cpu paramater fields
    %
    %
    % Common Neuron nomenclature by cell name, firing type, genetic markers
    % cell name (Py: pyramidal, Bs: basket, Mb: martinoti, Ch: chandellier
    % firing type (Rs: regular spiking (with adaptation), Fs: fast spiker
    % Nad: regular non addapting)
    % genetic markers (Pv: parvalbumin, Som: somatostatin ...
    
    
    
    properties
        %% general construction paramaters
        
        nNeurons        %the number of neurons being simulated
        cellTypes        %1xnNeuronTypes cellarray identifying the cell types, currently not a necessary variable but useful for book keeping
        percCellTypes   %1xnNeuronTypes array between 0 and 1 and suming to 1 representing the percentage of the network for each cell type
        
        %time constant for the simulation which for now is actually hardcoaded as 1 so this is just a placeholder
        
        aMean %a itz neuron paramater 1xnCellTypes distributions are gaussian with this mean
        aVar    %a itz neuron paramater 1xnCellTypes distributions are gaussian with this variance
        dMean %a itz neuron paramater 1xnCellTypes distributions are gaussian with this mean
        dVar    %a itz neuron paramater 1xnCellTypes distributions are gaussian with this variance
        
        %% input paramaters
        
        inputMode = 'Thal';  %type of input in the simulation thalamic pre pool 'Thal', spontanious mini EPSPs 'Mini' (not supported yet),
        
        nInputs;        %size of the pool presynaptic neurons when inputMode = 'Thal'
        inputTime_ms;        %the amount of time in ms the presynaptic neurons are active active when inputMode = 'Thal' used to speed up the simulation
        inputWeightMat;           %the synaptic weights from the presynaptic neurons to the network when inputMode = 'Thal'       
        inputSpikes;             %1xnInputs cell array of the input spiketimes
        
        
        
        %% general simulation paramaters
        
        useGPU = 0; %simulation flag, 0 is no GPU simulation
        useCPU = 0; %simulation flag, 0 is no CPU simulation
        
        threshold = 40;        %this is still hardcoded for now
        seconds2simulate;    %amount of time to simulate
        dt = 1;
        
        %% CPU paramaters for simulation
        
        %static variables
        
        synapticTau;     %nNeurons by nNeurons sparce matix of static tause for the synapses used in CPU simulation
        weightMat;  %nNeurons by nNeurons sparce matix of static weights for the synapses used in CPU simulation
        axonDelayMat;   %nNeurons by nNeurons sparce matix of static delays for the synapses used in CPU simulation
        a
        d
        
        %dynamic variables
        
        synapseMat;     %nNeurons by nNeurons sparce matix of dynamic state of the synapses while simulation on the CPU
        axonState;     %nNeurons by nNeurons sparce matix of dynamic state of the axons while simulation on the CPU
        I
        v
        u
        
        %% GPU paramaters for simulation on a GPU
        
        %for the kernal and GPU variables set up

        paramMatGPU; %5xnConnection matrix, row 1 a list of presynaptic neurons, row 2 a post synaptic neurons, row 3 synaptic weights, row 4 axonDelay, row 5 synapticTau     
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

        This should work really well with a sparse generating matrix?
        %}
        
        nConnections %the number of connections between neurons needed for GPU simulations
        
        %static variables on the GPU 
        
        nrnPreIndexArray_gpu;
        nrnPostIndexArray_gpu;
        postSynapseList_gpu;
        axonDelay_gpu;   %1xnConnections static delays for the synapses used in GPU simulation
        synapseWeight_gpu;  %1xnConnections static weights for the synapses used in GPU simulation
        tau_gpu;     %1xnConnections static tause for the synapses used in GPU simulation
        a_gpu
        d_gpu
        dt_gpu
        
        %dynamic variables on the GPU 
        
        axonState_gpu;  %1xnConnections dynamic state of the axons while simulation on the GPU
        synapseState_gpu;   %1xnConnections dynamic state of the synapses while simulation on the GPU
        spikeNeuron_gpu;    %1xnNeurons ('boolian') array dynamicly updated to 1 if the neurons fires or 0 if it didnt in the GPU simulation
        v_gpu
        u_gpu
        I_gpu
            
        %% results paramaters
        patchIndx
        patch
        spikes
        
        %% general class paramaters
        verbose = 0;       %flag used to specify if reporting lines get printed or not
    end
    
    properties (SetAccess = private)
        
        nNeuronTypes      %the number of neuron types
        nNeuronsByType   %1xnNeuronTypes array suming to nNeurons representing the number of cells for each cell type. this is compleatly defined by percCellTypes and nNeurons
        
   end
    
    methods
        
        function itzSimulationObj = itzSimulation
        end
        
        function check = paramaterCheck(itzSimulationObj)
            %check neuron number consistancy
            check = sum(itzSimulationObj.nNeuronsByType) == itzSimulationObj.nNeuron;
            %check cell type consistance
            check = check*length(unique(itzSimulationObj.nNeuronTypes,length(itzSimulationObj.nNeuronsByType),length(itzSimulationObj.percCellTypes))) == 1;
        end
        
        function itzSimulationObj = buildNeurons(itzSimulationObj)
            %sets a,d, for each cell type, as well as nNeuronTypes. requires nNeurons, percCellTypes, aVar,aMean, dVar, dVar
            %this must be run before building any connectivity structures
            
            itzSimulationObj.nNeuronTypes = length(itzSimulationObj.percCellTypes);
            
            itzSimulationObj.nNeuronsByType(1) = ceil(itzSimulationObj.nNeurons*itzSimulationObj.percCellTypes(1));
            for iNeuronType = 2:length(itzSimulationObj.percCellTypes)
                itzSimulationObj.nNeuronsByType(iNeuronType) = floor(itzSimulationObj.nNeurons*itzSimulationObj.percCellTypes(iNeuronType));
            end
            
            %make sure there are nNeurons total despite proportions
            itzSimulationObj.nNeuronsByType(1) = itzSimulationObj.nNeurons - sum( itzSimulationObj.nNeuronsByType(2:end));
                
            if itzSimulationObj.useGPU
                
                as = zeros(1,itzSimulationObj.nNeurons);
                 startIndx = cumsum([1,itzSimulationObj.nNeuronsByType]);
                for iNeuronType = 1:length(itzSimulationObj.nNeuronsByType)
                    as(startIndx(iNeuronType):(startIndx(iNeuronType+1)-1)) = itzSimulationObj.aMean(iNeuronType)+randn(1,itzSimulationObj.nNeuronsByType(iNeuronType))*itzSimulationObj.aVar(iNeuronType);
                end
                itzSimulationObj.a_gpu =  gpuArray(single(as));
                clear as

                ds = zeros(1,itzSimulationObj.nNeurons);
                for iNeuronType = 1:length(itzSimulationObj.nNeuronsByType)
                    ds(startIndx(iNeuronType):(startIndx(iNeuronType+1)-1)) = itzSimulationObj.dMean(iNeuronType)+randn(1,itzSimulationObj.nNeuronsByType(iNeuronType))*itzSimulationObj.dVar(iNeuronType);
                end
                itzSimulationObj.d_gpu =  gpuArray(single(ds));
                clear ds

            else
                
                itzSimulationObj.a = zeros(1,itzSimulationObj.nNeurons);
                startIndx = cumsum([1,itzSimulationObj.nNeuronsByType]);
                for iNeuronType = 1:length(itzSimulationObj.nNeuronsByType)
                    itzSimulationObj.a(startIndx(iNeuronType):(startIndx(iNeuronType+1)-1)) = itzSimulationObj.aMean(iNeuronType)+randn(1,itzSimulationObj.nNeuronsByType(iNeuronType))*itzSimulationObj.aVar(iNeuronType);
                end
                
                itzSimulationObj.d = zeros(1,itzSimulationObj.nNeurons);
                for iNeuronType = 1:length(itzSimulationObj.nNeuronsByType)
                    itzSimulationObj.d(startIndx(iNeuronType):(startIndx(iNeuronType+1)-1)) = itzSimulationObj.dMean(iNeuronType)+randn(1,itzSimulationObj.nNeuronsByType(iNeuronType))*itzSimulationObj.dVar(iNeuronType);
                end
                
            end
            
        end
        
        function itzSimulationObj = runSimulation(itzSimulationObj)
            
            if itzSimulationObj.useGPU < 1
                 itzSimulationObj = runCPUSimulation(itzSimulationObj);
            else
                itzSimulationObj = runSimulationGPU(itzSimulationObj);
            end
            
        end
        

    end
    
end

