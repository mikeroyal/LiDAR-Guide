<h1 align="center">
 <img src="https://user-images.githubusercontent.com/45159366/121950812-f5ae2900-cd0e-11eb-8989-9188bd18a68c.png">
  <br />
  LiDAR Guide
</h1>

#### A guide covering LiDAR including the applications, libraries and tools that will make you better and more efficient with LiDAR development.

 **Note: You can easily convert this markdown file to a PDF in [VSCode](https://code.visualstudio.com/) using this handy extension [Markdown PDF](https://marketplace.visualstudio.com/items?itemName=yzane.markdown-pdf).**

1. [LiDAR Learning Resources](https://github.com/mikeroyal/LiDAR-Guide#lidar-learning-resources)

2. [LiDAR Tools & Frameworks](https://github.com/mikeroyal/LiDAR-Guide#lidar-tools--frameworks)

3. [Machine Learning](https://github.com/mikeroyal/LiDAR-Guide#machine-learning)

4. [CUDA Development](https://github.com/mikeroyal/LiDAR-Guide#cuda-development)

5. [Robotics](https://github.com/mikeroyal/LiDAR-Guide#robotics)

6. [MATLAB Development](https://github.com/mikeroyal/LiDAR-Guide#matlab-development)

 <p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/121950830-fb0b7380-cd0e-11eb-8b58-c23706b33c29.png">
  <br />
</p>

# LiDAR Learning Resources

[Introduction to Lidar Course - NOAA](https://coast.noaa.gov/digitalcoast/training/intro-lidar.html)

[Lidar 101:An Introduction to Lidar Technology, Data, and Applications(PDF) - NOAA](https://coast.noaa.gov/data/digitalcoast/pdf/lidar-101.pdf)

[Understanding LiDAR Technologies - GIS Lounge](https://www.gislounge.com/understanding-lidar-technologies/)

[LiDAR University Free Lidar Training Courses on MODUS AI](https://www.modus-ai.com/lidar-university-2/)

[LiDAR | Learning Plan on ERSI](https://www.esri.com/training/catalog/5bccd52a6e9c0f01fb49e85d/lidar/#!)

[Light Detection and Ranging Sensors Course on Coursera](https://www.coursera.org/lecture/state-estimation-localization-self-driving-cars/lesson-1-light-detection-and-ranging-sensors-3NXgp)

[Quick Introduction to Lidar and Basic Lidar Tools(PDF)](https://training.fws.gov/courses/references/tutorials/geospatial/CSP7304/documents/Lidar.pdf)

[LIDAR - GIS Wiki](http://wiki.gis.com/wiki/index.php/Lidar)

[OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Main_Page)

[OpenStreetMap Frameworks](https://wiki.openstreetmap.org/wiki/Frameworks)

# LiDAR Tools & Frameworks
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

[Light Detection and Ranging (lidar)](https://www.usgs.gov/news/earthword-lidar) is a technology used to create high-resolution models of ground elevation with a vertical accuracy of 10 centimeters (4 inches). Lidar equipment, which includes a laser scanner, a Global Positioning System (GPS), and an Inertial Navigation System (INS), is typically mounted on a small aircraft. The laser scanner transmits brief pulses of light to the ground surface. Those pulses are reflected or scattered back and their travel time is used to calculate the distance between the laser scanner and the ground.  Lidar data is initially collected as a “point cloud” of individual points reflected from everything on the surface, including structures and vegetation. To produce a “bare earth” Digital Elevation Model (DEM), structures and vegetation are stripped away.

 <p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/121950840-fe9efa80-cd0e-11eb-9a12-57c4799d63b5.png">
  <br />
</p>

**3D Data Visualization of Golden Gate Bridge. Source: [USGS](https://www.usgs.gov/core-science-systems/ngp/tnm-delivery)**

[Mola](https://docs.mola-slam.org/latest/) is a Modular Optimization framework for Localization and mApping (MOLA).

 <p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/121950850-01015480-cd0f-11eb-9fa6-1f93d6d87cd1.gif">
  <br />
</p>

**3D LiDAR SLAM from KITTI dataset. Source: [MOLA](https://docs.mola-slam.org/latest/demo-kitti-lidar-slam.html)**

[Lidar Toolbox™](https://www.mathworks.com/products/lidar.html) is a MATLAB tool that provides algorithms, functions, and apps for designing, analyzing, and testing lidar processing systems. You can perform object detection and tracking, semantic segmentation, shape fitting, lidar registration, and obstacle detection. Lidar Toolbox supports lidar-camera cross calibration for workflows that combine computer vision and lidar processing.

[Automated Driving Toolbox™](https://www.mathworks.com/products/automated-driving.html) is a MATLAB tool that provides algorithms and tools for designing, simulating, and testing ADAS and autonomous driving systems. You can design and test vision and lidar perception systems, as well as sensor fusion, path planning, and vehicle controllers. Visualization tools include a bird’s-eye-view plot and scope for sensor coverage, detections and tracks, and displays for video, lidar, and maps. The toolbox lets you import and work with HERE HD Live Map data and OpenDRIVE® road networks. It also provides reference application examples for common ADAS and automated driving features, including FCW, AEB, ACC, LKA, and parking valet. The toolbox supports C/C++ code generation for rapid prototyping and HIL testing, with support for sensor fusion, tracking, path planning, and vehicle controller algorithms.

[Microsoft AirSim](https://microsoft.github.io/AirSim/lidar.html) is a simulator for drones, cars and more, built on Unreal Engine (with an experimental Unity release). AirSim is open-source, cross platform, and supports [software-in-the-loop simulation](https://www.mathworks.com/help///ecoder/software-in-the-loop-sil-simulation.html) with popular flight controllers such as PX4 & ArduPilot and [hardware-in-loop](https://www.ni.com/en-us/innovations/white-papers/17/what-is-hardware-in-the-loop-.html) with PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped into any Unreal environment. AirSim is being developed  as a platform for AI research to experiment with deep learning, computer vision and reinforcement learning algorithms for autonomous vehicles.

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/121950853-02328180-cd0f-11eb-9459-1b31d084bd3f.png">
  <br />
</p>

**3D Autonomous Vehicle Simulation in AirSim. Source: [Microsoft](https://microsoft.github.io/AirSim)**

[LASer(LAS)](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities) is a public file format for the interchange of 3-dimensional point cloud data data between data users. Although developed primarily for exchange of lidar point cloud data, this format supports the exchange of any 3-dimensional x,y,z tuplet. This binary file format is an alternative to proprietary systems or a generic ASCII file interchange system used by many companies. The problem with proprietary systems is obvious in that data cannot be easily taken from one system to another. There are two major problems with the ASCII file interchange. The first problem is performance because the reading and interpretation of ASCII elevation data can be very slow and the file size can be extremely large even for small amounts of data. The second problem is that all information specific to the lidar data is lost. The LAS file format is a binary file format that maintains information specific to the lidar nature of the data while not being overly complex.

[3D point cloud](https://www.onyxscan-lidar.com/point-cloud/) is a set of data points defined in a given three-dimensional coordinates system.. Point clouds can be produced directly by 3D scanner which records a large number of points returned from the external surfaces of objects or earth surface. These data are exchanged between LiDAR users mainly through LAS format files (.las).

[ArcGIS Desktop](https://www.esri.com/en-us/arcgis/products/arcgis-desktop/overview) is powerful and cost-effective desktop geographic information system (GIS) software. It is the essential software package for GIS professionals. ArcGIS Desktop users can create, analyze, manage, and share geographic information so decision-makers can make intelligent, informed decisions.

[USGS 3DEP Lidar Point Cloud Now Available as Amazon Public Dataset](https://www.usgs.gov/news/usgs-3dep-lidar-point-cloud-now-available-amazon-public-dataset)

[National Geospatial Program](https://www.usgs.gov/core-science-systems/national-geospatial-program)

[National Map Data Download and Visualization Services](https://www.usgs.gov/core-science-systems/ngp/tnm-delivery)

[USGS Lidar Base Specification(LBS) online edition](https://www.usgs.gov/core-science-systems/ngp/ss/lidar-base-specification-online)

# Machine Learning
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/96352527-ad077880-1078-11eb-98b7-da1c0586cf0e.png">
  <br />
</p>

<img src="https://user-images.githubusercontent.com/45159366/105645196-dccfd480-5e4e-11eb-95d1-c5eb560b72fd.jpeg">

**Machine Learning/Deep Learning Frameworks.**

## Learning Resources for ML

[Machine Learning](https://www.ibm.com/cloud/learn/machine-learning) is a branch of artificial intelligence (AI) focused on building apps using algorithms that learn from data models and improve their accuracy over time without needing to be programmed.

[Machine Learning by Stanford University from Coursera](https://www.coursera.org/learn/machine-learning)

[AWS Training and Certification for Machine Learning (ML) Courses](https://aws.amazon.com/training/learning-paths/machine-learning/)

[Machine Learning Scholarship Program for Microsoft Azure from Udacity](https://www.udacity.com/scholarships/machine-learning-scholarship-microsoft-azure)

[Microsoft Certified: Azure Data Scientist Associate](https://docs.microsoft.com/en-us/learn/certifications/azure-data-scientist)

[Microsoft Certified: Azure AI Engineer Associate](https://docs.microsoft.com/en-us/learn/certifications/azure-ai-engineer)

[Azure Machine Learning training and deployment](https://docs.microsoft.com/en-us/azure/devops/pipelines/targets/azure-machine-learning)

[Learning Machine learning and artificial intelligence from Google Cloud Training](https://cloud.google.com/training/machinelearning-ai)

[Machine Learning Crash Course for Google Cloud](https://developers.google.com/machine-learning/crash-course/)

[JupyterLab](https://jupyterlab.readthedocs.io/)

[Scheduling Jupyter notebooks on Amazon SageMaker ephemeral instances](https://aws.amazon.com/blogs/machine-learning/scheduling-jupyter-notebooks-on-sagemaker-ephemeral-instances/)

[How to run Jupyter Notebooks in your Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-jupyter-notebooks)

[Machine Learning Courses Online from Udemy](https://www.udemy.com/topic/machine-learning/)

[Machine Learning Courses Online from Coursera](https://www.coursera.org/courses?query=machine%20learning&)

[Learn Machine Learning with Online Courses and Classes from edX](https://www.edx.org/learn/machine-learning)

## ML Frameworks, Libraries, and Tools

[TensorFlow](https://www.tensorflow.org) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

[Keras](https://keras.io) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.It was developed with a focus on enabling fast experimentation. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML.

[PyTorch](https://pytorch.org) is a library for deep learning on irregular input data such as graphs, point clouds, and manifolds. Primarily developed by Facebook's AI Research lab.

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models.

[Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/) is a fast and collaborative Apache Spark-based big data analytics service designed for data science and data engineering. Azure Databricks, sets up your Apache Spark environment in minutes, autoscale, and collaborate on shared projects in an interactive workspace. Azure Databricks supports Python, Scala, R, Java, and SQL, as well as data science frameworks and libraries including TensorFlow, PyTorch, and scikit-learn.

[Microsoft Cognitive Toolkit (CNTK)](https://docs.microsoft.com/en-us/cognitive-toolkit/) is an open-source toolkit for commercial-grade distributed deep learning. It describes neural networks as a series of computational steps via a directed graph. CNTK allows the user to easily realize and combine popular model types such as feed-forward DNNs, convolutional neural networks (CNNs) and recurrent neural networks (RNNs/LSTMs). CNTK implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers.

[Apple CoreML](https://developer.apple.com/documentation/coreml) is a framework that helps integrate machine learning models into your app. Core ML provides a unified representation for all models. Your app uses Core ML APIs and user data to make predictions, and to train or fine-tune models, all on the user's device. A model is the result of applying a machine learning algorithm to a set of training data. You use a model to make predictions based on new input data.

[Tensorflow_macOS](https://github.com/apple/tensorflow_macos) is a Mac-optimized version of TensorFlow and TensorFlow Addons for macOS 11.0+ accelerated using Apple's ML Compute framework.

[Apache OpenNLP](https://opennlp.apache.org/) is an open-source library for a machine learning based toolkit used in the processing of natural language text. It features an API for use cases like [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition), [Sentence Detection](), [POS(Part-Of-Speech) tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging), [Tokenization](https://en.wikipedia.org/wiki/Tokenization_(data_security)) [Feature extraction](https://en.wikipedia.org/wiki/Feature_extraction), [Chunking](https://en.wikipedia.org/wiki/Chunking_(psychology)), [Parsing](https://en.wikipedia.org/wiki/Parsing), and [Coreference resolution](https://en.wikipedia.org/wiki/Coreference).

[Apache Airflow](https://airflow.apache.org) is an open-source workflow management platform created by the community to programmatically author, schedule and monitor workflows. Install. Principles. Scalable. Airflow has a modular architecture and uses a message queue to orchestrate an arbitrary number of workers. Airflow is ready to scale to infinity.

[Open Neural Network Exchange(ONNX)](https://github.com/onnx) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

[Apache MXNet](https://mxnet.apache.org/) is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines. Support for Python, R, Julia, Scala, Go, Javascript and more.

[AutoGluon](https://autogluon.mxnet.io/index.html) is toolkit for Deep learning that automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications. With just a few lines of code, you can train and deploy high-accuracy deep learning models on tabular, image, and text data.

[Anaconda](https://www.anaconda.com/) is a very popular Data Science platform for machine learning and deep learning that enables users to develop models, train them, and deploy them.

[PlaidML](https://github.com/plaidml/plaidml) is an advanced and portable tensor compiler for enabling deep learning on laptops, embedded devices, or other devices where the available computing hardware is not well supported or the available software stack contains unpalatable license restrictions.

[OpenCV](https://opencv.org) is a highly optimized library with focus on real-time computer vision applications. The C++, Python, and Java interfaces support Linux, MacOS, Windows, iOS, and Android.

[Scikit-Learn](https://scikit-learn.org/stable/index.html) is a Python module for machine learning built on top of SciPy, NumPy, and matplotlib, making it easier to apply robust and simple implementations of many popular machine learning algorithms.

[Weka](https://www.cs.waikato.ac.nz/ml/weka/) is an open source machine learning software that can be accessed through a graphical user interface, standard terminal applications, or a Java API. It is widely used for teaching, research, and industrial applications, contains a plethora of built-in tools for standard machine learning tasks, and additionally gives transparent access to well-known toolboxes such as scikit-learn, R, and Deeplearning4j.

[Caffe](https://github.com/BVLC/caffe) is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR)/The Berkeley Vision and Learning Center (BVLC) and community contributors.

[Theano](https://github.com/Theano/Theano) is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently including tight integration with NumPy.

[nGraph](https://github.com/NervanaSystems/ngraph) is an open source C++ library, compiler and runtime for Deep Learning. The nGraph Compiler aims to accelerate developing AI workloads using any deep learning framework and deploying to a variety of hardware targets.It provides the freedom, performance, and ease-of-use to AI developers.

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for [deep neural networks](https://developer.nvidia.com/deep-learning). cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. cuDNN accelerates widely used deep learning frameworks, including [Caffe2](https://caffe2.ai/), [Chainer](https://chainer.org/), [Keras](https://keras.io/), [MATLAB](https://www.mathworks.com/solutions/deep-learning.html), [MxNet](https://mxnet.incubator.apache.org/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/).

[Jupyter Notebook](https://jupyter.org/) is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Jupyter is used widely in industries that do data cleaning and transformation, numerical simulation, statistical modeling, data visualization, data science, and machine learning.

[Apache Spark](https://spark.apache.org/) is a unified analytics engine for large-scale data processing. It provides high-level APIs in Scala, Java, Python, and R, and an optimized engine that supports general computation graphs for data analysis. It also supports a rich set of higher-level tools including Spark SQL for SQL and DataFrames, MLlib for machine learning, GraphX for graph processing, and Structured Streaming for stream processing.

[Apache Spark Connector for SQL Server and Azure SQL](https://github.com/microsoft/sql-spark-connector) is a high-performance connector that enables you to use transactional data in big data analytics and persists results for ad-hoc queries or reporting. The connector allows you to use any SQL database, on-premises or in the cloud, as an input data source or output data sink for Spark jobs.

[Apache PredictionIO](https://predictionio.apache.org/) is an open source machine learning framework for developers, data scientists, and end users. It supports event collection, deployment of algorithms, evaluation, querying predictive results via REST APIs. It is based on scalable open source services like Hadoop, HBase (and other DBs), Elasticsearch, Spark and implements what is called a Lambda Architecture.

[Cluster Manager for Apache Kafka(CMAK)](https://github.com/yahoo/CMAK) is a tool for managing [Apache Kafka](https://kafka.apache.org/) clusters.

[BigDL](https://bigdl-project.github.io/) is a distributed deep learning library for Apache Spark. With BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.

[Eclipse Deeplearning4J (DL4J)](https://deeplearning4j.konduit.ai/) is a set of projects intended to support all the needs of a JVM-based(Scala, Kotlin, Clojure, and Groovy) deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks.

[Tensorman](https://github.com/pop-os/tensorman) is a utility for easy management of Tensorflow containers by developed by [System76]( https://system76.com).Tensorman allows Tensorflow to operate in an isolated environment that is contained from the rest of the system. This virtual environment can operate independent of the base system, allowing you to use any version of Tensorflow on any version of a Linux distribution that supports the Docker runtime.

[Numba](https://github.com/numba/numba) is an open source, NumPy-aware optimizing compiler for Python sponsored by Anaconda, Inc. It uses the LLVM compiler project to generate machine code from Python syntax. Numba can compile a large subset of numerically-focused Python, including many NumPy functions. Additionally, Numba has support for automatic parallelization of loops, generation of GPU-accelerated code, and creation of ufuncs and C callbacks.

[Chainer](https://chainer.org/) is a Python-based deep learning framework aiming at flexibility. It provides automatic differentiation APIs based on the define-by-run approach (dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.

[XGBoost](https://xgboost.readthedocs.io/) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. It supports distributed training on multiple machines, including AWS, GCE, Azure, and Yarn clusters. Also, it can be integrated with Flink, Spark and other cloud dataflow systems.

[cuML](https://github.com/rapidsai/cuml) is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects. cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. In most cases, cuML's Python API matches the API from scikit-learn.

# CUDA Development
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/94306481-e17b8f00-ff27-11ea-832f-c85374acb3b1.png">
  <br />
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/117718735-55a23480-b191-11eb-874d-e690d09cd490.png">
  <br />
</p>

**CUDA Toolkit. Source: [NVIDIA Developer CUDA](https://developer.nvidia.com/cuda-zone)**

**[Point Cloud Processing with NVIDIA DriveWorks SDK](https://developer.nvidia.com/blog/point-cloud-processing-nvidia-driveworks-sdk/)**

## CUDA Learning Resources

[CUDA](https://developer.nvidia.com/cuda-zone) is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs. In GPU-accelerated applications, the sequential part of the workload runs on the CPU, which is optimized for single-threaded. The compute intensive portion of the application runs on thousands of GPU cores in parallel. When using CUDA, developers can program in popular languages such as C, C++, Fortran, Python and MATLAB.

[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)

[CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

[CUDA GPU support for TensorFlow](https://www.tensorflow.org/install/gpu)

[NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)

[NVIDIA GPU Cloud Documentation](https://docs.nvidia.com/ngc/ngc-introduction/index.html)

[NVIDIA NGC](https://ngc.nvidia.com/) is a hub for GPU-optimized software for deep learning, machine learning, and high-performance computing (HPC) workloads.

[NVIDIA NGC Containers](https://www.nvidia.com/en-us/gpu-cloud/containers/) is a registry that provides researchers, data scientists, and developers with simple access to a comprehensive catalog of GPU-accelerated software for AI, machine learning and HPC. These containers take full advantage of NVIDIA GPUs on-premises and in the cloud.

## CUDA Tools

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is a collection of tools & libraries that provide a development environment for creating high performance GPU-accelerated applications. The CUDA Toolkit allows you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to build and deploy your application on major architectures including x86, Arm and POWER.

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for [deep neural networks](https://developer.nvidia.com/deep-learning). cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. cuDNN accelerates widely used deep learning frameworks, including [Caffe2](https://caffe2.ai/), [Chainer](https://chainer.org/), [Keras](https://keras.io/), [MATLAB](https://www.mathworks.com/solutions/deep-learning.html), [MxNet](https://mxnet.incubator.apache.org/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/).

[CUDA-X HPC](https://www.nvidia.com/en-us/technologies/cuda-x/) is a collection of libraries, tools, compilers and APIs that help developers solve the world's most challenging problems. CUDA-X HPC includes highly tuned kernels essential for high-performance computing (HPC).

[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is a collection of tools & libraries that allows users to build and run GPU accelerated Docker containers. The toolkit includes a container runtime [library](https://github.com/NVIDIA/libnvidia-container) and utilities to automatically configure containers to leverage NVIDIA GPUs.

[Minkowski Engine](https://nvidia.github.io/MinkowskiEngine) is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors.

[CUTLASS](https://github.com/NVIDIA/cutlass) is a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) at all levels and scales within CUDA. It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS.

[CUB](https://github.com/NVIDIA/cub) is a cooperative primitives for CUDA C++ kernel authors.

[Tensorman](https://github.com/pop-os/tensorman) is a utility for easy management of Tensorflow containers by developed by [System76]( https://system76.com).Tensorman allows Tensorflow to operate in an isolated environment that is contained from the rest of the system. This virtual environment can operate independent of the base system, allowing you to use any version of Tensorflow on any version of a Linux distribution that supports the Docker runtime.

[Numba](https://github.com/numba/numba) is an open source, NumPy-aware optimizing compiler for Python sponsored by Anaconda, Inc. It uses the LLVM compiler project to generate machine code from Python syntax. Numba can compile a large subset of numerically-focused Python, including many NumPy functions. Additionally, Numba has support for automatic parallelization of loops, generation of GPU-accelerated code, and creation of ufuncs and C callbacks.

[Chainer](https://chainer.org/) is a Python-based deep learning framework aiming at flexibility. It provides automatic differentiation APIs based on the define-by-run approach (dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.

[CuPy](https://cupy.dev/) is an implementation of NumPy-compatible multi-dimensional array on CUDA. CuPy consists of the core multi-dimensional array class, cupy.ndarray, and many functions on it. It supports a subset of numpy.ndarray interface.

[CatBoost](https://catboost.ai/) is a fast, scalable, high performance [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) on Decision Trees library, used for ranking, classification, regression and other machine learning tasks for Python, R, Java, C++. Supports computation on CPU and GPU.

[cuDF](https://rapids.ai/) is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data. cuDF provides a pandas-like API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows without going into the details of CUDA programming.

[cuML](https://github.com/rapidsai/cuml) is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects. cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. In most cases, cuML's Python API matches the API from scikit-learn.

[ArrayFire](https://arrayfire.com/) is a general-purpose library that simplifies the process of developing software that targets parallel and massively-parallel architectures including CPUs, GPUs, and other hardware acceleration devices.

[Thrust](https://github.com/NVIDIA/thrust) is a C++ parallel programming library which resembles the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs.

[AresDB](https://eng.uber.com/aresdb/) is a GPU-powered real-time analytics storage and query engine. It features low query latency, high data freshness and highly efficient in-memory and on disk storage management.

[Arraymancer](https://mratsim.github.io/Arraymancer/) is a tensor (N-dimensional array) project in Nim. The main focus is providing a fast and ergonomic CPU, Cuda and OpenCL ndarray library on which to build a scientific computing ecosystem.

[Kintinuous](https://github.com/mp3guy/Kintinuous) is a real-time dense visual SLAM system capable of producing high quality globally consistent point and mesh reconstructions over hundreds of metres in real-time with only a low-cost commodity RGB-D sensor.

[GraphVite](https://graphvite.io/) is a general graph embedding engine, dedicated to high-speed and large-scale embedding learning in various applications.


# Robotics
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/96352533-b55fb380-1078-11eb-874c-f165cbcce899.png">
  <br />
</p>

**[Accelerating Lidar for Robotics with NVIDIA CUDA-based PCL](https://developer.nvidia.com/blog/accelerating-lidar-for-robotics-with-cuda-based-pcl/)**

## Robotics Learning Resources

[Robotics courses from Coursera](https://www.edx.org/learn/robotics)

[Learn Robotics with Online Courses and Classes from edX](https://www.edx.org/learn/robotics)

[Top Robotics Courses Online from Udemy](https://www.udemy.com/topic/robotics/)

[Free Online AI & Robotics Courses](https://www.futurelearn.com/subjects/it-and-computer-science-courses/ai-and-robotics)

[REC Foundation Robotics Industry Certification](https://www.roboticseducation.org/industry-certifications/)

[Carnegie Mellon Robotics Academy](https://www.cmu.edu/roboticsacademy/Training/Certifications.html)

[RIA Robotic Integrator Certification Program](https://www.robotics.org/robotics/integrator-certification)

[AWS RoboMaker – Develop, Test, Deploy, and Manage Intelligent Robotics Apps](https://aws.amazon.com/blogs/aws/aws-robomaker-develop-test-deploy-and-manage-intelligent-robotics-apps/)

[Microsoft AI School](https://aischool.microsoft.com/en-us/home)

[Language Understanding (LUIS) for Azure Cognitive Services](https://docs.microsoft.com/en-us/azure/cognitive-services/luis/what-is-luis)

[ROS on Windows 10](https://ms-iot.github.io/ROSOnWindows/)

[Windows ML ROS Node](https://ms-iot.github.io/ROSOnWindows/ROSAtMS/WinML.html)

[Azure VM templates to bootstrap ROS and ROS 2 environments](https://ms-iot.github.io/ROSOnWindows/ROSAtMS/AzureVM.html)

[Google Robotics Research](https://research.google/teams/brain/robotics/)

## Tools for Robotics

[Robot Framework](https://robotframework.org/) is a generic open source automation framework. It can be used for test automation and robotic process automation. It has easy syntax, utilizing human-readable keywords. Its capabilities can be extended by libraries implemented with Python or Java.

[The Robotics Library (RL)](https://github.com/roboticslibrary/rl) is a self-contained C++ library for robot kinematics, motion planning and control. It covers mathematics, kinematics and dynamics, hardware abstraction, motion planning, collision detection, and visualization.RL runs on many different systems, including Linux, macOS, and Windows. It uses CMake as a build system and can be compiled with Clang, GCC, and Visual Studio.

[Robot Structural Analysis Professional](https://www.autodesk.com/products/robot-structural-analysis/overview?term=1-YEAR) is structural load analysis software developed by Autodesk that verifies code compliance and uses BIM-integrated workflows to exchange data with Revit. It can help you to create more resilient, constructible designs that are accurate, coordinated, and connected to BIM.

[PowerMill](https://www.autodesk.com/products/powermill/overview) is a software developed by Autodesk that provides powerful, flexible, easy-to-use tools for offline programming of robots. Get tools to help you optimize robotic paths and simulate virtual mock-ups of manufacturing cells and systems.

[ROS](https://www.ros.org/) is robotics middleware. Although ROS is not an operating system, it provides services designed for a heterogeneous computer cluster such as hardware abstraction, low-level device control, implementation of commonly used functionality, message-passing between processes, and package management.

[ROS2](https://index.ros.org/doc/ros2/) is a set of [software libraries and tools](https://github.com/ros2) that help you build robot applications. From drivers to state-of-the-art algorithms, and with powerful developer tools, ROS has what you need for your next robotics project. And it’s all open source.

[MoveIt](https://moveit.ros.org/) is the most widely used software for manipulation and has been used on over 100 robots. It provides an easy-to-use robotics platform for developing advanced applications, evaluating new designs and building integrated products for industrial, commercial, R&D, and other domains.

[AutoGluon](https://autogluon.mxnet.io/index.html) is toolkit for [Deep learning](https://gitlab.com/maos20008/intro-to-machine-learning) that automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications. With just a few lines of code, you can train and deploy high-accuracy deep learning models on tabular, image, and text data.

[Gazebo](http://gazebosim.org/) accurately and efficiently simulates indoor and outdoor robots. You get a robust physics engine, high-quality graphics, and programmatic and graphical interfaces.

[Robotics System Toolbox](https://www.mathworks.com/products/robotics.html) provides tools and algorithms for designing, simulating, and testing manipulators, mobile robots, and humanoid robots. For manipulators and humanoid robots, the toolbox includes algorithms for collision checking, trajectory generation, forward and inverse kinematics, and dynamics using a rigid body tree representation.
For mobile robots, it includes algorithms for mapping, localization, path planning, path following, and motion control. The toolbox provides reference examples of common industrial robot applications. It also includes a library of
commercially available industrial robot models that you can import, visualize, and simulate.

[Intel Robot DevKit](https://github.com/intel/robot_devkit) is the tool to generate Robotics Software Development Kit (RDK) designed for autonomous devices, including the ROS2 core and capacibilities packages like perception, planning, control driver etc. It provides flexible build/runtime configurations to meet different autonomous requirement on top of diversity hardware choices, for example use different hareware engine CPU/GPU/VPU to accelerate AI related features.

[Arduino](https://www.arduino.cc/) is an open-source platform used for building electronics projects. Arduino consists of both a physical programmable circuit board (often referred to as a microcontroller) and a piece of software, or IDE (Integrated Development Environment) that runs on your computer, used to write and upload computer code to the physical board.

[ArduPilot](https://ardupilot.org/ardupilot/index.html) enables the creation and use of trusted, autonomous, unmanned vehicle systems for the peaceful benefit of all. ArduPilot provides a comprehensive suite of tools suitable for almost any vehicle and application.

[AirSim](https://github.com/Microsoft/AirSim) is a simulator for drones, cars and more, built on Unreal Engine (we now also have an experimental Unity release). It is open-source, cross platform, and supports hardware-in-loop with popular flight controllers such as PX4 for physically and visually realistic simulations.

[The JPL Open Source Rover](https://github.com/nasa-jpl/open-source-rover) is an open source, build it yourself, scaled down version of the 6 wheel rover design that JPL uses to explore the surface of Mars. The Open Source Rover is designed almost entirely out of consumer off the shelf (COTS) parts. This project is intended to be a teaching and learning experience for those who want to get involved in mechanical engineering, software, electronics, or robotics.

[Light Detection and Ranging(LiDAR)](https://en.wikipedia.org/wiki/Lidar) is a remote sensing method that uses light in the form of a pulsed laser at an object, and uses the time and wavelength of the reflected beam of light to estimate the distance and in some applications ([Laser Imaging](https://en.wikipedia.org/wiki/Laser_scanning)), to create a 3D representation of the object and its surface characteristics. This technology is commonly used in aircraft and self-driving vehicles.

[AliceVision](https://github.com/alicevision/AliceVision) is a Photogrammetric Computer Vision Framework which provides a 3D Reconstruction and Camera Tracking algorithms. AliceVision aims to provide strong software basis with state-of-the-art computer vision algorithms that can be tested, analyzed and reused. The project is a result of collaboration between academia and industry to provide cutting-edge algorithms with the robustness and the quality required for production usage.

[CARLA](https://github.com/carla-simulator/carla) is an open-source simulator for autonomous driving research. CARLA has been developed from the ground up to support development, training, and validation of autonomous driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites and environmental conditions.

[ROS bridge](https://github.com/carla-simulator/ros-bridge) is a package to bridge ROS for CARLA Simulator.

[ROS-Industrial](https://rosindustrial.org/) is an open source project that extends the advanced capabilities of ROS software to manufacturing.

[AWS RoboMaker](https://aws.amazon.com/robomaker/) is the most complete cloud solution for robotic developers to simulate, test and securely deploy robotic applications at scale. RoboMaker provides a fully-managed, scalable infrastructure for simulation that customers use for multi-robot simulation and CI/CD integration with regression testing in simulation.

[Microsoft Robotics Developer Studio](https://www.microsoft.com/en-us/download/details.aspx?id=29081)  is a free .NET-based programming environment for building robotics applications.

[Visual Studio Code Extension for ROS](https://github.com/ms-iot/vscode-ros) is an extension provides support for Robot Operating System (ROS) development.

[Azure Kinect ROS Driver](https://github.com/microsoft/azure_kinect_ros_driver) is a node which publishes sensor data from the [Azure Kinect Developer Kit](https://azure.microsoft.com/en-us/services/kinect-dk/) to the [Robot Operating System (ROS)](http://www.ros.org/). Developers working with ROS can use this node to connect an Azure Kinect Developer Kit to an existing ROS installation.

[Azure IoT Hub for ROS](https://github.com/microsoft/ros_azure_iothub) is a ROS package works with the Microsoft Azure IoT Hub service to relay telemetry messages from the Robot to Azure IoT Hub or reflect properties from the Digital Twin to the robot using dynamic reconfigure.

[ROS 2 with ONNX Runtime](https://github.com/ms-iot/ros_msft_onnx) is a program that uses ROS 2 to run on different hardware platforms using their respective AI acceleration libraries for optimized execution of the ONNX model.

[Azure Cognitive Services LUIS ROS Node](https://github.com/ms-iot/ros_msft_luis) is a ROS node that bridges between ROS and the Azure Language Understanding Service. it can be configured to process audio directly from a microphone, or can subscribe to a ROS audio topic, then processes speech and generates "intent" ROS messages which can be processed by another ROS node to generate ROS commands.

# MATLAB Development
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

<p align="center">
 <img src="https://user-images.githubusercontent.com/45159366/94306473-de809e80-ff27-11ea-924b-0a6947ae38bc.png">
  <br />
</p>

**[Lidar Toolbox - MATLAB](https://www.mathworks.com/products/lidar.html)**

**[Lidar Processing - MATLAB & Simulink](https://www.mathworks.com/help/driving/lidar-processing.html)**

**[Lidar Toolbox Documentation - MATLAB & Simulink](https://www.mathworks.com/help/lidar/index.html)**

**[Automated Driving Toolbox - MATLAB](https://www.mathworks.com/products/automated-driving.html)**

**[Getting Started with Lidar Acquisition in MATLAB Video](https://www.mathworks.com/videos/getting-started-with-lidar-acquisition-in-matlab-1563511401517.html)**

**[Point Cloud Processing - MATLAB & Simulink](https://www.mathworks.com/help/vision/lidar-and-point-cloud-processing.html)**

## MATLAB Learning Resources

[MATLAB](https://www.mathworks.com/products/matlab.html) is a programming language that does numerical computing such as expressing matrix and array mathematics directly.

[MATLAB Documentation](https://www.mathworks.com/help/matlab/)

[Getting Started with MATLAB ](https://www.mathworks.com/help/matlab/getting-started-with-matlab.html)

[MATLAB and Simulink Training from MATLAB Academy](https://matlabacademy.mathworks.com)

[MathWorks Certification Program](https://www.mathworks.com/services/training/certification.html)

[MATLAB Online Courses from Udemy](https://www.udemy.com/topic/matlab/)

[MATLAB Online Courses from Coursera](https://www.coursera.org/courses?query=matlab)

[MATLAB Online Courses from edX](https://www.edx.org/learn/matlab)

[Building a MATLAB GUI](https://www.mathworks.com/discovery/matlab-gui.html)

[MATLAB Style Guidelines 2.0](https://www.mathworks.com/matlabcentral/fileexchange/46056-matlab-style-guidelines-2-0)

[Setting Up Git Source Control with MATLAB & Simulink](https://www.mathworks.com/help/matlab/matlab_prog/set-up-git-source-control.html)

[Pull, Push and Fetch Files with Git with MATLAB & Simulink](https://www.mathworks.com/help/matlab/matlab_prog/push-and-fetch-with-git.html)

[Create New Repository with MATLAB & Simulink](https://www.mathworks.com/help/matlab/matlab_prog/add-folder-to-source-control.html)

[PRMLT](http://prml.github.io/) is Matlab code for machine learning algorithms in the PRML book.

## MATLAB Tools

[MATLAB Online](https://matlab.mathworks.com) allows to users to uilitize MATLAB and Simulink through a web browser such as Google Chrome.

[Simulink](https://www.mathworks.com/products/simulink.html) is a block diagram environment for Model-Based Design. It supports simulation, automatic code generation, and continuous testing of embedded systems.

[MATLAB Schemer](https://github.com/scottclowe/matlab-schemer) is a MATLAB package makes it easy to change the color scheme (theme) of the MATLAB display and GUI.

[LRSLibrary](https://github.com/andrewssobral/lrslibrary) is a Low-Rank and Sparse Tools for Background Modeling and Subtraction in Videos. The library was designed for moving object detection in videos, but it can be also used for other computer vision and machine learning problems.

[Robotics Toolbox for MATLAB](https://www.mathworks.com/products/robotics.html) provides a toolbox that brings robotics specific functionality(designing, simulating, and testing manipulators, mobile robots, and humanoid robots) to MATLAB, exploiting the native capabilities of MATLAB (linear algebra, portability, graphics). The toolbox also supports mobile robots with functions for robot motion models (bicycle), path planning algorithms (bug, distance transform, D*, PRM), kinodynamic planning (lattice, RRT), localization (EKF, particle filter), map building (EKF) and simultaneous localization and mapping (EKF), and a Simulink model a of non-holonomic vehicle. The Toolbox also including a detailed Simulink model for a quadrotor flying robot.

[SEA-MAT](https://sea-mat.github.io/sea-mat/) is a collaborative effort to organize and distribute Matlab tools for the Oceanographic Community.

[Gramm](https://github.com/piermorel/gramm) is a complete data visualization toolbox for Matlab. It provides an easy to use and high-level interface to produce publication-quality plots of complex data with varied statistical visualizations. Gramm is inspired by R's ggplot2 library.

[hctsa](https://hctsa-users.gitbook.io/hctsa-manual) is a software package for running highly comparative time-series analysis using Matlab.

[Plotly](https://plot.ly/matlab/) is a Graphing Library for MATLAB.

[YALMIP](https://yalmip.github.io/) is a MATLAB toolbox for optimization modeling.

[GNU Octave](https://www.gnu.org/software/octave/) is a high-level interpreted language, primarily intended for numerical computations. It provides capabilities for the numerical solution of linear and nonlinear problems, and for performing other numerical experiments. It also provides extensive graphics capabilities for data visualization and manipulation.

## Contribute

- [x] If would you like to contribute to this guide simply make a [Pull Request](https://github.com/mikeroyal/LiDAR-Guide/pulls).


## License
[Back to the Top](https://github.com/mikeroyal/LiDAR-Guide#table-of-contents)

Distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) Public License](https://creativecommons.org/licenses/by/4.0/).
