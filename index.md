# LISTIT Homepage 

LISTIT (Learning non-Isomorph Structured Transductions for Image and Text fragments) is a 4-years research project funded by the Italian Ministry of Education and Research, under the SIR framework (contract n.  RBSI14STDE). This is the official repository of the project referencing software libraries and code developed within the project and the associated pubblications. 

## Project objectives

The high level goal of LISTIT is the design and development of machine learning and deep learning methodologies generalizing supervised learning to structured samples both in input and output to the learning model. LISTIT considers primarily tree data but also targets more general classes of structures, including graphs with cycles. 

LISTIT applications target natural language processing, image captioning, biomedical and life-sciences data. 

## Models and Methodologies

LISTIT has built a range of learning models targeting
- **Structure embedding** : encoding of topology-varying input samples (trees, graphs) into fixed-size adaptive vectorial embeddings.
- **Structure decoding** : generation of topology-varying structured predictions (trees, graphs), possibly conditioned on vectorial encodings of input structures.

LISTIT methodology is based on a generative approach, mixing both probabilistic and neural (deep) learning models. Here follows a summary of models developed and released within the project, together with associated code (when available). Technical details about the different models can be found in the project bibliography in the dedicated section.

### Probabilistic recursive models for trees 
- Input-Output Bottom-up Hidden Tree Markov Model (IO-BHTMM) - A model learning a distribution over tree couples that serves to address isomorphic tree transduction problems, implemented throughout a generative process acting from the leaves to the root of the trees ([code](https://github.com/diningphil/IOBHTMM))
- Bayesian Mixture of Bottom-up Hidden Tree Markov Models (INF-BHTMM) - A non-parametric generative model extending the IO-BHTMM to deal with a mixture of potentially infinite probabilistic tree encoders ([code](https://gitlab.itc.unipi.it/d.castellana/Mixtures_SP_BHTMM)). 

### Recursive neural models for trees 
-  Tree2Tree Conditional Variational Autoencoder (T2TVAE) - T2TVAE is a Conditional Variational Autoencoder that can learn a generative process realizing non-isomorph tree transductions, by extending the VAE framework to allow conditioning the generative process on tree-structured inputs and to generate tree-structured predictions ([code](https://github.com/m-colombo/conditional-variational-tree-autoencoder)).
- Child-Sum-TreeLSTM (CS-TreeLSTM) - Baseline recursive bottom-up tree encoder assuming full stationariety. Used to encode trees where the position of nodes (ordering) with respect to their siblings is not relevantfor the task. Distributed within the [LISTIT Toolkit](https://github.com/Ant-Bru/Tree2TreeLearning) 
- NAry-TreeLSTM (NAry-TreeLSTM) - Baseline recursive bottom-up tree encoder assuming positional stationariety. Allows to discriminate children by their position with respect to the siblings. Distributed within the [LISTIT Toolkit](https://github.com/Ant-Bru/Tree2TreeLearning) 
- NAry-TreeLSTM (NAry-TreeLSTM) - Baseline recursive bottom-up tree encoder assuming positional stationariety. Allows to discriminate children by their position with respect to the siblings. Distributed within the [LISTIT Toolkit](https://github.com/Ant-Bru/Tree2TreeLearning)
- Sequential-TreeLSTM (seq-TreeLSTM) - Bottom-up tree encoder leveraging sequential children state aggregators with a choice of GRU, BidirectionalGRU and DoubleGRU models. Distributed within the [LISTIT Toolkit](https://github.com/Ant-Bru/Tree2TreeLearning)

### Tensor-based models for trees 
- Bayesian Tensor Factorisation Bottom-up Hidden Tree Markov Models (TF-BHTMM) - An tensor-based extension of the bottom-up hidden Markov model for the encoding of tree-structured data, allowing scalable higher-order state transition functions within a fully Bayesian framework ([code](https://gitlab.itc.unipi.it/d.castellana/TF_bhtmm)).
- Tensor Tree Recursive Neural Network (TTRNN) - A general framework extending Recursive Neural Networks for tree embedding with more expressive context aggregation functions leveraging tensor decompositions. Research code ([here](https://github.com/danielecastellana22/tensor-tree-nn)). Consolidated TTRNN models are also integrated and distributed through the [LISTIT Toolkit](https://github.com/Ant-Bru/Tree2TreeLearning).

### Probabilistic models for graphs
- Contextual Graph Markov Model (CGMM) - Probabilistic approach to learning contexts in graphs, combining information diffusion and local computation through the use of a deep architecture and stationarity assumptions ([code](https://github.com/diningphil/CGMM)).  

### Graph generation models

### Other models and contributions
- Hidden Tree Markov Network (HTN) - A neuro-probabilistic hybrid model for tree structured data processing combining heterogeneous probabilistic tree models within a neural architecture, trained by backpropagation ([code](https://github.com/vdecaro/Hidden-Tree-Markov-Network)). 
- Fair Graph Classification - A benchmarking suite to support fair assessment of learning models for graphs in classification tasks ([code](https://github.com/diningphil/gnn-comparison)). 

## Project Contributors and Collaborators

## The LISTIT Toolkit library

The tree-to-tree transduction framework at the core of the LISTIT project is available as a PYTORCH library, integrating various probabilistic and neural tree encoders and tree sampling models (decoders) developed within the project. The library is released, documented and maintaned [here](https://github.com/Ant-Bru/Tree2TreeLearning) and implements the following features:
- Neural tree encoders: recursive neural networks with a variety of aggregation functions (described in related papers), including fully stationary (ChildSum) and positional (N-ary) aggregation, and sequential aggregators leveraging Gru, BidirectionalGru and DoubleGru layers
- Tensor-decomposition encoders: neural and probabilistic models leveraging tensor-decomposition for higher-order childre-to-parent state aggregation
- Neural decoders: neural-based architectures for tree generation leveraging different decoding strategies, including N-ary decoding, chidren-by-children decoding, sequential children decoding.
- Tree-to-tree machine translation application

An efficient Tensorflow implementation of the Tree2Tree Conditional Variational Autoencoder (T2TVAE) has also been realized for enhanced computational efficiency, to deal with larger scale learning tasks (including Machine Translation and Image Captioning). The T2TVAE model implementation is released and maintaned [here](https://github.com/m-colombo/conditional-variational-tree-autoencoder). A version of the T2TVAE for image captioning applications is released [here](https://github.com/dave94-42/image_captionig_tree2tree), together with [code](https://github.com/dave94-42/image_captionig_tree2tree_input-target_processing) to generate tree-structured representations of visual content leveraging the PBM segmentation framework.  

## Project Publications
All project publications are freely accessible either as open access (when available, on the publisher site) or as pre-print. The list is under continuous update.

### Under review and in preparation
-	Davide Bacciu; Federico Errica; Alessio Micheli, Probabilistic Learning on Graphs via Contextual Architectures, Journal paper under review
- Davide Bacciu; Federico Errica; Alessio Micheli; Marco Podda; A Gentle Introduction to Deep Learning for Graphs, Journal paper under review (pdf)
- Andrea Valenti; Antonio Carta; Davide Bacciu; Learning a Latent Space of Style-Aware Music Representations by Adversarial Autoencoders, Submitted to conference, 2020
- Davide Bacciu; Daniele Castellana, Generalising Recursive Neural Models by Tensor Decomposition, Submitted to conference, 2020
- Davide Bacciu; Alessio Conte; Roberto Grossi; Francesco Landolfi; Andrea Marino, K-plex Pooling for Graph Neural Networks, Submitted to conference, 2020
- Federico Errica; Davide Bacciu; Alessio Micheli; Theoretically Expressive and Edge-aware Graph Learning, Submitted to conference, 2020
- Marco Podda; Davide Bacciu; Alessio Micheli; Paolo Milazzo; Biochemical Pathway Robustness Prediction with Graph Neural Networks, Submitted to conference, 2020

### Main project publications
- Davide Bacciu; Alessio Micheli; Marco Podda, Edge-based sequential graph generation with recurrent neural networks, Neurocomputing, 2020 (In press)
- Marco Podda; Davide Bacciu; Alessio Micheli; A Deep Generative Model for Fragment-Based Molecule Generation; Proceedings of 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), 2020
- Federico Errica; Marco Podda; Davide Bacciu; Alessio Micheli, A Fair Comparison of Graph Neural Networks for Graph Classification, Proceedings of the Eighth International Conference on Learning Representations (ICLR 2020), 2020
- Davide Bacciu; Daniele Castellana, Tensor Decompositions in Recursive Neural Networks for Tree-Structured Data, 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2020), 2020 (to appear)
- Davide Bacciu; Alessio Micheli; Deep Learning for Graphs, book chapter, Studies in Computational Intelligence Series, Springer, 2020 (to appear)
- Davide Bacciu; Daniele, Castellana, Bayesian Mixtures of Hidden Tree Markov Models for Structured Data Clustering, Neurocomputing, Vol. 342, 49-59, 2019
- Davide Bacciu; Luigi Di Sotto, A non-negative factorization approach to node pooling in graph convolutional neural networks, Proceedings of the 18th International Conference of the Italian Association for Artificial Intelligence (AIIA 2019), Lecture Notes in Artificial Intelligence, Vol. 11946, 294-306, Springer-Verlag, 2019
- Antonio Carta; Davide Bacciu, Sequential Sentence Embeddings for Semantic Similarity, Proceedings of the 2019 IEEE Symposium Series on Computational Intelligence (SSCI'19), 2019 (in Press)
- Daniele Castellana, Davide Bacciu, Bayesian Tensor Factorisation for Bottom-up Hidden Tree Markov Models, Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN 2019), IEEE, 2019.
- Davide Bacciu; Antonio Carta; Alessandro Sperduti, Linear Memory Networks, Proceedings of the 28th International Conference on Artificial Neural Networks (ICANN 2019), Lecture Notes in Computer Science Vol. 11727, 513-525, Springer-Verlag, 2019
- Davide Bacciu; Alessio Micheli; Marco Podda, Graph generation by sequential edge prediction, Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN'19), 95-100, i6doc.com, 2019.
-	Davide Bacciu; Antonio Bruno, Deep Tree Transductions - A Short Survey, Proceedings of the 2019 INNS Big Data and Deep Learning (INNS-BDDL 2019), Recent Advances in Big Data and Deep Learning, Vol.1, pp 236-245, Springer, 2019
-	Davide Bacciu; Federico, Errica;  Alessio Micheli, Contextual Graph Markov Model: A Deep and Generative Approach to Graph Processing, Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR, Vol. 80, 294-303, 2018.
-	Davide Bacciu; Alessio Micheli; Alessandro Sperduti “Generative Kernels for Tree-Structured Data” IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, vol. 29, no. 10, pp. 4932-4946, 2018
-	Davide Bacciu; Antonio Bruno, Text Summarization as Tree Transduction by Top-Down TreeLSTM, Proceedings of the 2018 IEEE Symposium Series on Computational Intelligence (SSCI'18), 1411-1418, IEEE, 2018
-	Davide Bacciu; Daniele Castellana, Learning Tree Distributions by Hidden Markov Models, Proceedings of the FLOC 2018 Workshop on Learning and Automata (LearnAut'18), 2018.
-	Davide Bacciu; Daniele Castellana, Mixture of Hidden Markov Models as Tree Encoder, Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN'18), 543-548, i6doc.com, 2018
-	Davide Bacciu. Hidden Tree Markov Networks: Deep and Wide Learning for Structured Data, Proceedings of the 2017 IEEE Symposium Series on Computational Intelligence (SSCI'17). IEEE, 2017
-	Davide Bacciu;  Claudio Gallicchio;  Alessio Micheli;  A reservoir activation kernel for trees, Proceedings  of the European Symposium on Artificial Neural Networks,  Computational Intelligence and Machine Learning (ESANN'16), 29-34, i6doc.com, 2016.

### Other contributed project publications
-	Davide, Bacciu; Francesco Crecchi, Augmenting Recurrent Neural Networks Resilience by Dropout, IEEE Transactions on Neural Networs and Learning Systems, 2019
-	Davide Bacciu;  Battista Biggio; Paulo J.G. Lisboa;  José D Martin;  Luca Oneto;  Alfredo Vellido, Societal Issues in Machine Learning: When Learning from Data is Not Enough, Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN'19), 455-464, i6doc.com, 2019.
-	Davide Bacciu; Andrea Bongiorno, Concentric ESN: Assessing the Effect of Modularity in Cycle Reservoirs, Proceedings of the 2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-8, 2018.
-	Davide Bacciu;  Paulo JG Lisboa;  Jose D Martin;  Ruxandra Stoean;  Alfredo Vellido, Bioinformatics and medicine in the era of deep learning, Proceedings  of the European Symposium on Artificial Neural Networks,  Computational Intelligence and Machine Learning (ESANN'18), 345-354, i6doc.com, 2018
-	Arapi, Visar; Santina, Cosimo Della; Bacciu, Davide; Bianchi, Matteo; Bicchi, Antonio, DeepDynamicHand: A deep neural architecture for labeling hand manipulation strategies in video sources exploiting temporal information, Frontiers in Neurorobotics, 12 , pp. 86, 2018.
-	Marco, Podda; Davide, Bacciu; Alessio, Micheli; Roberto, Bellu; Giulia, Placidi; Luigi, Gagliardi, A machine learning approach to estimating preterm infants survival: development of the Preterm Infants Survival Assessment (PISA) predictor, Nature Scientific Reports, 8, 2018
-	Davide Bacciu; Michele Colombo; Davide Morelli; David Plans “Randomized neural networks for preference learning with physiological data.” NEUROCOMPUTING, 298 , pp. 9-20, 2018
-	Davide Bacciu; Francesco Crecchi; Davide Morelli, DropIn: Making reservoir computing neural networks robust to missing inputs by dropout. Proceedings of the 2017 International Joint Conference on Neural Networks. p. 2080-2087, IEEE, 2017

## Contact Info

For further information about the project please contact the PI [Davide Bacciu](http://www.di.unipi.it/~bacciu/).

For aspects related to code and software, please use the contact information provided for the single repositories. 
