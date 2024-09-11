# This is meant to be a notebook of my experiments
The idea is to use the notebook to recrord arbitrarty hyperparameter choices in the face of teaching a new neural network architectrure. Hopefully it will supply a guidance that wish to use my code in the future.

## Preamble
For the record, this is a series of experiments I have had with the intentention of predicting the of deactivated CasRx in gene translation within prokaryotes, specifically E. coli in this case. The remaining lines are a realization of the computational expirements with which I have derived my conclusions. Here I will paste my experimentaion and results for record.

## Experimental Procedures
The data that is used for training here is derived from an oligo based cloning technique which is effectictive in creating CRISPR gRNA vectors. The resulting plasmids can be used to target specific genes. The dataset used is focused on the is available in the Github repository and is focused on a 3nt tilling of GFP. The architecthure in graphical form will be placed later.

## September 5th 2024
### Hyperparmeters 
#### Trial 1
    stride_parameter = 30 
    label_dim = EndHorizon-StartHorizon 
    embedding_dim = 18 
    batch_size_parameter = 20 
    n_pre_post_layers = 10
    outpuDim = EndHorizon-StartHorizon
    feedforwardDepth = 5
    feedforwardDim = 30
    intermediate_dim = 50
    debug_splash = 0
    this_step_size_val = 0.01
    this_max_iters = 2e6
#### Results
    Reconstruction Loss: 0.0026666094
    Embedding Loss: 0.015375573
    step 417000 , validation error 0.393819
    step 417000 , test error 0.029194
    Reconstruction Loss: 0.0029331816
    Embedding Loss: 0.015318816
## September 6th
### Trial one
### Hyperparmaters
    StartHorizon = 100 # these are in unit of timepoints not time
    EndHorizon= 300r
    stride_parameter = 30 #Determinded by the length of the gRNA sequence
    label_dim = EndHorizon-StartHorizon # To match the number of timepoints in the plate reader data
    embedding_dim = 18 #18 was a good dimension for embedding Alec's gRNA sequences that resulted in near perfect reconstruction 
    batch_size_parameter = 20
    n_pre_post_layers = 10
    outpuDim = EndHorizon-StartHorizon
    feedforwardDepth = 5
    feedforwardDim = 30
    intermediate_dim = 50 
    debug_splash = 0
    this_step_size_val = 0.01
    this_max_iters = 2e6
    this_corpus,this_labels = make_labeled_corpus(allseqs, data, stride_parameter)
### Results 
These hyperparameters were used last night but take a while to run due to the large max iters. 
The next change that needs to be explored is to create a loss function which considers both the Embeddign loss and the regression loss.

The results from this took alot of time and are as follows.