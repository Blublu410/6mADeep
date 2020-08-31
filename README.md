# 6mADeep
# 6mADeep: N6-methyladenine sites prediction with an interpretable model

N6-methyladenine (6mA) DNA  modification carries abundant epigenetic information and  plays a crucial role in biological functions, which is of great significance to the development of human beings. Accurate and reliable methods for detecting 6mA sites can help understand the mechanism of DNA modification. Current methods for identifying sites generally depend on experiments, but these are time-consuming and expensive. Under this circumstance, a variety of  computer-based approaches emerge.


## Requirements
1. Python2
2. Anaconda 
3. keras
4. numpy, sklearn, pickle, pandas, matplotlib

## 6mA datasets preparation
Please prepare the 6mA dataset containing the positive and negative samples in **CSV** file format.
Each CSV file contains 2 columns : DNA sequence (41bp long), label (0 means negative, 1 means positive), separated by comma.

For the positive CSV file, the example is below:

![pos](https://github.com/Blublu410/6mADeep/blob/master/Image/csvPos.PNG)


For the negative CSV file, the example is below:

![neg](https://github.com/Blublu410/6mADeep/blob/master/Image/csvNeg.PNG)


For the convenience, we upload 2 6mA datasets for rice genome proposed by previous studies **(Chen et al., 2019)**.

The 6mA-rice-Chen dataset is in the folder **6mA_data/Rice_Chen**. The 6mA-rice-Chen dataset contains 880 positive samples and 880 negative samples. DNA sequences in both positive samples and negative samples are 41bp long. For each positive sequence, its center is the 6mA modification site. For each negative sequence, its center contains no 6mA modification site. 

## Train the model on the 6mA dataset
There are two model sciprts in the folder 6mADeep/, including 6mADeep.py,and 6mADeep-K.py.

- The 6mADeep.py is the training code for the basic model based on the genomic sequence only.
- The 6mADeep-K.py is the training code for the model without XGBoost algorithm.

We will show the method to use these scripts based on the Chen dataset **(Chen et al., 2019)**.

### Model training and analyzing

For the 6mADeep.py, the usage method is in below:
```
python 6mADeep.py [OPTIONS]
[OPTIONS]:
	--positive     positive 6mA csv file
	--negative     negative 6mA csv file
	--output       output folder
```

For the 6mADeep-K.py, the usage method is in below:
```
python 6mADeep-K.py [OPTIONS]
[OPTIONS]:
	--positive     positive 6mA csv file
	--negative     negative 6mA csv file
	--output       output folder
```


### Model result
The model result in the output folder, contains the following things.

```
	performance.txt     statistics of metrics of training, validation and testing set
	model(*).h5         model files because of the cross-validation
	ROC.png             ROC cure of training, validation and testing set
```

For example:
![result](https://github.com/Blublu410/6mADeep/blob/master/Image/Result.PNG) 
![result2](https://github.com/Blublu410/6mADeep/blob/master/Image/ROC.PNG)

### Model evaluation
Use the script, 6mAEvaluate.py, in 6mADeep/ to evaluate the model

When using the training code to build the model on the dataset, there are multiple models because of cross-validation. These trained models are somewhat similar, and users can choose any of them.

```
python evaluate.py [OPTIONS]
[OPTIONS]:
	--h5File     the model h5 file 
	--csv       the input csv file 
```
 The script will output the accuracy of the model on the input dataset 

## Examples of using the program

### 1 training and anazlying on the 6mA-rice-Chen by 6mADeep.py

Using the following commands:
```
python 6mADeep.py --positive Dataset/Rice_Chen/Positive.txt --negative Dataset/Rice_Chen/Negative.txt --output result/
```

### 2 training and anazlying on the 6mA-rice-Chen by 6mADeep-K.py

Using the following commands:
```
python 6mADeep-K.py --positive Dataset/Rice_Chen/Positive.txt --negative Dataset/Rice_Chen/Negative.txt --output result/
```


## Reference
1. Chen, W., Lv, H., Nie, F., and Lin, H. (2019). i6mA-Pred: Identifying DNA N6-methyladenine sites in the rice genome. Bioinformatics.

