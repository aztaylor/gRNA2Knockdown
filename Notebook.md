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
    step 404000 , validation error 0.281793
    step 404000 , test error 0.030925
    Reconstruction Loss: 0.004284358
    Embedding Loss: 0.0154286055
    step 405000 , validation error 0.310379
    step 405000 , test error 0.0298142
    Reconstruction Loss: 0.0027975051
    Embedding Loss: 0.015460129
    step 406000 , validation error 0.325973
    step 406000 , test error 0.0269577
    Reconstruction Loss: 0.0024939657
    Embedding Loss: 0.015397804
    step 407000 , validation error 0.343879
    step 407000 , test error 0.0292902
    Reconstruction Loss: 0.0044097276
    Embedding Loss: 0.0155791715
    step 408000 , validation error 0.294723
    step 408000 , test error 0.029043
    Reconstruction Loss: 0.0028405609
    Embedding Loss: 0.01550237
    step 409000 , validation error 0.295991
    step 409000 , test error 0.030089
    Reconstruction Loss: 0.003567892
    Embedding Loss: 0.015696166
    step 410000 , validation error 0.289261
    step 410000 , test error 0.0278034
    Reconstruction Loss: 0.0022184022
    Embedding Loss: 0.015595641
    step 411000 , validation error 0.310479
    step 411000 , test error 0.0292406
    Reconstruction Loss: 0.0028185998
    Embedding Loss: 0.015428416
    step 412000 , validation error 0.301879
    step 412000 , test error 0.0294907
    Reconstruction Loss: 0.0025039
    Embedding Loss: 0.015551361
    step 413000 , validation error 0.302791
    step 413000 , test error 0.0271914
    Reconstruction Loss: 0.0033832088
    Embedding Loss: 0.015350331
    step 414000 , validation error 0.298497
    step 414000 , test error 0.0279061
    Reconstruction Loss: 0.003218405
    Embedding Loss: 0.015468246
    step 415000 , validation error 0.290696
    step 415000 , test error 0.0273089
    Reconstruction Loss: 0.0022478974
    Embedding Loss: 0.015403614
    step 416000 , validation error 0.300739
    step 416000 , test error 0.0278266
    Reconstruction Loss: 0.0026666094
    Embedding Loss: 0.015375573
    step 417000 , validation error 0.393819
    step 417000 , test error 0.029194
    Reconstruction Loss: 0.0029331816
    Embedding Loss: 0.015318816

