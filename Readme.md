# Welcome to vacation2020 program																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				

  

  This is a program for Dachuang and design of graduation. 

  The project is about BirdClef . As for now, I have finished the model and preprocessing.

  Here's some   problems that should be handling following:

1.  how to deal with the audio that get many species make sound as same time? (***cocktail party problem***)
2.  Cause it's vacation time. The model cannot be too huge (though I can train the model with cpu), I still hope to decrease the data dimension.(Dose Image can do that?)
3. Transfer learning seems like a good idea for data unbalance. The method based on model is easy, but I want use some method like TCA, JCA that based on dataset's features. Not only for transfer learning, but also for decreasing the dataset's size.
4. ***The problems above come out from my mind right now, not last...***

***2020/2/27***

Uploading the basic code including data preprocessing, data generator, model.

**2020/3/2**

Have trained three models which are deep model, overfitting model, residual net respectively.

But the trend like a straight line, I will focus on fixing it. and improve the GPU efficiency is also important.

**2020/3/3**

I have trained ten models and I get two points:

- the neural network I have made can get the point of accuracy is about 54% (So the preprocessing is import )
- I should train a model with a lot of epochs until overfit, otherwise I cannot see whether the model is get the best point.