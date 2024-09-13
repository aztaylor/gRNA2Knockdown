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

### Nov. 1 3th 
U normalized loss results in an early plateu

These hyperparameters were used last night but take a while to run due to the large max iters. 
The next change that needs to be explored is to create a loss function which considers both the Embeddign loss and the regression loss.

The results from this took alot of time and are as follows.
2024-09-13 14:49:43.716023: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2024-09-13 14:49:43.767093: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2024-09-13 14:49:43.767563: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-09-13 14:49:44.587869: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
listed fold change data shape: (96, 200)
['TTGATCTCCTTTTTAAGTGAACTTGGGCCC', 'TTTCATTGTTGATCTCCTTTTTAAGTGAAC', 'AAAATTGCTTTCATTGTTGATCTCCTTTTT', 'TCAGTACGAAAATTGCTTTCATTGTTGATC', 'AAGATGTTTCAGTACGAAAATTGCTTTCAT', 'GCATGATTAAGATGTTTCAGTACGAAAATT', 'CCTCCCCTGCATGATTAAGATGTTTCAGTA', 'TTAGAAACCCTCCCCTGCATGATTAAGATG', 'AAGCTCCATTAGAAACCCTCCCCTGCATGA', 'CCAGTGAAAAGCTCCATTAGAAACCCTCCC', 'GAACAACGCCAGTGAAAAGCTCCATTAGAA', 'CAGGATGGGAACAACGCCAGTGAAAAGCTC', 'AGCTCGACCAGGATGGGAACAACGCCAGTG', 'CGCCGTCCAGCTCGACCAGGATGGGAACAA', 'GTTTACGTCGCCGTCCAGCTCGACCAGGAT', 'TTGTGGCCGTTTACGTCGCCGTCCAGCTCG', 'CGCTGAACTTGTGGCCGTTTACGTCGCCGT', 'GCCGGACACGCTGAACTTGTGGCCGTTTAC', 'TCGCCCTCGCCGGACACGCTGAACTTGTGG', 'CATCGCCCTCGCCCTCGCCGGACACGCTGA', 'GTAGGTGGCATCGCCCTCGCCCTCGCCGGA', 'AGCTTGCCGTAGGTGGCATCGCCCTCGCCC', 'TCAGGGTCAGCTTGCCGTAGGTGGCATCGC', 'GATGAACTTCAGGGTCAGCTTGCCGTAGGT', 'GTGGTGCAGATGAACTTCAGGGTCAGCTTG', 'GCTTGCCGGTGGTGCAGATGAACTTCAGGG', 'CACGGGCAGCTTGCCGGTGGTGCAGATGAA', 'GGCCAGGGCACGGGCAGCTTGCCGGTGGTG', 'CGAGGGTGGGCCAGGGCACGGGCAGCTTGC', 'GGTGGTCACGAGGGTGGGCCAGGGCACGGG', 'TAGGTCAGGGTGGTCACGAGGGTGGGCCAG', 'GCACGCCGTAGGTCAGGGTGGTCACGAGGG', 'GAAGCACTGCACGCCGTAGGTCAGGGTGGT', 'TAGCGGCTGAAGCACTGCACGCCGTAGGTC', 'GGTCGGGGTAGCGGCTGAAGCACTGCACGC', 'CTTCATGTGGTCGGGGTAGCGGCTGAAGCA', 'TCGTGCTGCTTCATGTGGTCGGGGTAGCGG', 'TGAAGAAGTCGTGCTGCTTCATGTGGTCGG', 'GGCGGACTTGAAGAAGTCGTGCTGCTTCAT', 'TCGGGCATGGCGGACTTGAAGAAGTCGTGC', 'CGTAGCCTTCGGGCATGGCGGACTTGAAGA', 'CTCCTGGACGTAGCCTTCGGGCATGGCGGA', 'ATGGTGCGCTCCTGGACGTAGCCTTCGGGC', 'TGAAGAAGATGGTGCGCTCCTGGACGTAGC', 'GTCGTCCTTGAAGAAGATGGTGCGCTCCTG', 'TAGTTGCCGTCGTCCTTGAAGAAGATGGTG', 'GGGTCTTGTAGTTGCCGTCGTCCTTGAAGA', 'CTCGGCGCGGGTCTTGTAGTTGCCGTCGTC', 'AACTTCACCTCGGCGCGGGTCTTGTAGTTG', 'CGCCCTCGAACTTCACCTCGGCGCGGGTCT', 'CAGGGTGTCGCCCTCGAACTTCACCTCGGC', 'CGGTTCACCAGGGTGTCGCCCTCGAACTTC', 'GCTCGATGCGGTTCACCAGGGTGTCGCCCT', 'GCCCTTCAGCTCGATGCGGTTCACCAGGGT', 'AAGTCGATGCCCTTCAGCTCGATGCGGTTC', 'CCTCCTTGAAGTCGATGCCCTTCAGCTCGA', 'GTTGCCGTCCTCCTTGAAGTCGATGCCCTT', 'CCCAGGATGTTGCCGTCCTCCTTGAAGTCG', 'GCTTGTGCCCCAGGATGTTGCCGTCCTCCT', 'GTACTCCAGCTTGTGCCCCAGGATGTTGCC', 'TTGTAGTTGTACTCCAGCTTGTGCCCCAGG', 'TGTGGCTGTTGTAGTTGTACTCCAGCTTGT', 'ATAGACGTTGTGGCTGTTGTAGTTGTACTC', 'GCCATGATATAGACGTTGTGGCTGTTGTAG', 'GCTTGTCGGCCATGATATAGACGTTGTGGC', 'GTTCTTCTGCTTGTCGGCCATGATATAGAC', 'TTGATGCCGTTCTTCTGCTTGTCGGCCATG', 'AGTTCACCTTGATGCCGTTCTTCTGCTTGT', 'GATCTTGAAGTTCACCTTGATGCCGTTCTT', 'TTGTGGCGGATCTTGAAGTTCACCTTGATG', 'CCTCGATGTTGTGGCGGATCTTGAAGTTCA', 'GCTGCCGTCCTCGATGTTGTGGCGGATCTT', 'AGCTGCACGCTGCCGTCCTCGATGTTGTGG', 'GGTCGGCGAGCTGCACGCTGCCGTCCTCGA', 'CTGGTAGTGGTCGGCGAGCTGCACGCTGCC', 'GTGTTCTGCTGGTAGTGGTCGGCGAGCTGC', 'CGATGGGGGTGTTCTGCTGGTAGTGGTCGG', 'GCCGTCGCCGATGGGGGTGTTCTGCTGGTA', 'AGCACGGGGCCGTCGCCGATGGGGGTGTTC', 'CGGGCAGCAGCACGGGGCCGTCGCCGATGG', 'GTGGTTGTCGGGCAGCAGCACGGGGCCGTC', 'CTCAGGTAGTGGTTGTCGGGCAGCAGCACG', 'ACTGGGTGCTCAGGTAGTGGTTGTCGGGCA', 'CAGGGCGGACTGGGTGCTCAGGTAGTGGTT', 'TCTTTGCTCAGGGCGGACTGGGTGCTCAGG', 'CGTTGGGGTCTTTGCTCAGGGCGGACTGGG', 'GCGCTTCTCGTTGGGGTCTTTGCTCAGGGC', 'ATGTGATCGCGCTTCTCGTTGGGGTCTTTG', 'GCAGGACCATGTGATCGCGCTTCTCGTTGG', 'GAACTCCAGCAGGACCATGTGATCGCGCTT', 'GCGGTCACGAACTCCAGCAGGACCATGTGA', 'TCCCGGCGGCGGTCACGAACTCCAGCAGGA', 'TCATTAGATCCCGGCGGCGGTCACGAACTC', 'CTGAAAGTTCATTAGATCCCGGCGGCGGTC', 'TTTTTTGGCTGAAAGTTCATTAGATCCCGG', 'GTCTTAAGTTTTTTGGCTGAAAGTTCATTA']
2024-09-13 14:49:47.430068: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1960] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
n_depth: 22
n_depth: 6
IP_Matrix_y shape: TensorShape([None, 200])
IP_Matrix_e shape: TensorShape([None, 18])
IP_Matrix_y shape: TensorShape([None, None])
IP_Matrix_e shape: TensorShape([None, None])
WARNING:tensorflow:From /home/yeunglab/.local/lib/python3.8/site-packages/tensorflow/python/training/adagrad.py:138: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
2024-09-13 14:49:48.551975: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
IP_Matrix_y shape: TensorShape([None, 30])
IP_Matrix_e shape: TensorShape([None, 18])
IP_Matrix_y shape: TensorShape([None, None])
IP_Matrix_e shape: TensorShape([None, None])
 step 1 , validation error 5241
 step 1 , test error 5719.74
 Reconstruction Loss: 1.4341176
 Embedding Loss: 42772.742
 step 1000 , validation error 975.896
 step 1000 , test error 179.81
 Reconstruction Loss: 0.11110277
 Embedding Loss: 16822.088
 step 2000 , validation error 1094.33
 step 2000 , test error 87.7027
 Reconstruction Loss: 0.10278458
 Embedding Loss: 16381.95
 step 3000 , validation error 1088.94
 step 3000 , test error 113.335
 Reconstruction Loss: 0.1052346
 Embedding Loss: 17140.355
 step 4000 , validation error 1137.99
 step 4000 , test error 68.4143
 Reconstruction Loss: 0.10107815
 Embedding Loss: 16975.125