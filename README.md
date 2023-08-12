========== MINTL in Matlab ==========    
This is the code for our paper “Multi-instance Nonparallel Tube Learning”, 2023.    

Author: Yanshan Xiao    
Contact: xiaoyanshan@189.cn    
Version: 1.0    

# 1. Introduction
In this paper, we propose a multi-instance nonparallel tube learning (MINTL) method. For a K-class multi-instance classification problem, it learns K nonparallel tubes, each for one class. To construct the ϵk-tube of class k, each bag of class k should have at least one instance included in the ϵk-tube. Moreover, the remaining instances should lie inside the ϵk-tube or outside the ϵk-tube with a large margin. At last, the ϵk-tube should lie far from the other classes.    

# 2. Requirements
Matlab R2020a version    

# 3. Datasets      
    (1) MINTL is evaluated on several real-world multi-instance learning datasets, i.e., Corel, SIVAL, 20 Newsgroup, AWA-3, Scene and Flower.    
    a) The Corel dataset contains 20 classes. These classes are divided into 6 groups, and 6 sub-datasets are obtained, i.e., Corel1, Corel2, …, Corel6. Each sub-dataset has 3~4 classes. More details can refer to our paper.   
    b) The SIVAL dataset contains 25 classes. These classes are divided into 8 groups, and 8 sub-datasets are obtained, i.e., SIVAL1, SIVAL 2, …, SIVAL8.     
    c) The 20 Newsgroup dataset contains 20 classes. These classes are divided into 6 groups, and 6 sub-datasets are obtained, i.e., Newsgroup1, Newsgroup2, …, Newsgroup6.     
    d) The Scene dataset contains 5 classes.    
    e) The AWA-3 dataset contains 3 classes.    
    f) The Flower dataset contains 5 classes.    

    (2) For the above datasets, the data format is as follows:     
    Bag_id, feature 1, feature 2, …, feature m, label   

    (3) All the above experimental datasets can be downloaded from the “data” fold.    

# 4. Run codes
    (1)How to run the codes?
    In Matlab command line, type:
    main(trainData,testData,C,V,kertype,Delta)

    (2) Input data:
    trainData: the training data. 
    testData:  the testing data.
    The data format is as follows:
    Bag_id, feature1, feature2, …, feature m, label

    (3) kertype: ’linear’ or ’rbf’  
    a) kertype=’linear’: K(x,z)=x*z      
    Delta is a dummy variable and can be set to any values. They have no influence on the results.
    b) Kertype=’rbf’: K(x,z)=exp(-||x-z||^2/(2* Delta^2))
    Delta is the kernel parameter in the RBF kernel.

    (4) C,V: regularization parameters. More details can refer to our paper.    

    (5) “demo.m” gives an example of how to run our codes.    

# 5. Output   
The best testing accuracy, optimal parameter values and so on are saved in the “result.mat” file which lies in the “result” fold.    


For further any inquires, feel free to contact me at xiaoyanshan@189.cn or post an issue here.    
