'''
epoch=1 forward and backward pass of ALL traning samples.
batch size= No of training samples in one forward and backward pass
no of iterations= no of passes, each pass using [batch size] no of samples
e.g. 100 samples, batch size=20 --> 100/20=5 iterations for 1 epoch




To determine the number of iterations needed for one complete pass through the book (or an epoch), 
Epoch divided the total number of training samples (100) by the batch size (20). 
This calculation gave him the number of iterations required for each epoch. In this case, it was 5.
So, the training process began. Epoch took the first 20 training samples from the book and fed them to the model. 
The model read each sample, pondered over the question, and tried to generate an answer. 
This was called the forward pass, as the information flowed forward through the model.

After the forward pass, the model had to learn from its mistakes and adjust its internal parameters accordingly. 
Epoch performed a backward pass, where he analyzed the model's generated answers and compared them to the 
correct answers in the training samples. By understanding the errors, the model could improve its performance.

Epoch repeated this process for all 20 samples in the batch, correcting and updating the model's parameters along the way. 
When he completed one pass through the 20 samples, he considered it one iteration. The model had taken a step towards learning.

Epoch continued this process for the remaining four batches of 20 samples each, 
completing a total of five iterations. With each iteration, the model learned more and more from the training samples, 
gradually improving its ability to answer questions accurately.

After these five iterations, Epoch celebrated the completion of one epoch. But his quest wasn't over.
He knew that the model needed more epochs to become truly proficient. So, he decided to repeat the entire process, 
starting from the first batch and continuing until he completed multiple epochs.


'''

import torch
import torch.nn as nn
import torchvison
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

