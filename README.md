# Exploring Latent Dynamics for Visual Storytelling

This is the PyTorch implementation for inference and training of the plan-CVAE framework as described in:

> **Exploring Latent Dynamics for Visual Storytelling**

Inspired by the visual story creation process that humans always consider the influence of the next sentence before they continue the story writing, we address the story generation by learning an information dynamic plan in a latent space for every sentence.

<p align="center">
     <img src="images/case.png" alt="Latent Dynamics">
     <br/>
     <sub><em>
      Generating visual story conditioned on latent dynamics. 
    </em></sub>
</p>


## 1. Model Structure 

The intuition behind the plan-CVAE framework is to learn a latent space with smooth temporal dynamics for modeling and generating a coherent visual story. 
The following figure illustrates the generation process of a visual story generator guided by a latent dynamic module. 

<p align="center">
     <img src="images/framework.png" alt="plan-CVAE">
     <br/>
     <sub><em>
      Overview of the proposed plan-CVAE framework. 
    </em></sub>
</p>



## 2. Requirements

Clone the repository and install the package for environment using the `requirements.txt` file:
```
pip install -r requirements.txt
```


## 3. Dataset 
1. Download the visual storytelling dataset, e.g., [VIST](https://visionandlanguage.net/VIST/) and [VSPE](https://github.com/tingyaohsu/VIST-Edit), and put it under the data path ```./dataset```.
2. Preprocess data with the provided script and run: ```python dataset.py```. 



## 4. Training

To training the plan-CVAE model on the corresponding dataset, run:
```
bash train.sh
```

Script has several important hyper-parameters that you can play with:
- ```--train_data_path```: Path of the training dataset. 
- ```--val_data_path```: Path of the validation dataset. 
- ```--output_dir```: Path of the saving model. 


## 5. CDE Score Evaluation 

To evaluate the performance of visual storytelling systerm, run:
```
python CDE_score.py
```



## Contact Information
For help or issues related to this package, please submit a GitHub issue. 


