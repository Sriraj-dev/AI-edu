##Supervised Learning : 
'''
It truns out that even when you create a model , you may need to evaluate your model further to
test its performance and also may be to improve its performance and have an idea on 
what next need to be done to improve its performance.

May be 
- You can collect more training data
- You can manipulate with the features like increasing ,decreasing or polynomial features
- May be incereasing or decreasing the value of lambda.

So we need a proper evaluation procedure(Diagnostics) to decide on what needs to be done to 
improve its performance


You Can :
1. Split the training set into trainingSet and testSet, this helps you to evaluate the performance
of the model.

    Cost = Logistic/Linear Loss  +  Regularization loss
    minimise it and then evaluate
    - Cost_Train_set & Cost_Of_Test_set.

    In case of classification , you can also check out what fraction of examples are misclassified
    to get a glance on the performance of the model.
    Basically Jtrain is not the only estimate to judge your model but rather Jtest would be a better
    estimate on how the model will perform.


2. It comes out that there is another better method to acually fairly estimate your model.
    Split the data into - training set, Cross validation set, Testing set

    You may have multiple models that u might think will fit the data, you can train them using 
    training set.
    Calculate the Jcv of all these models and then u will be able to chose the model which gives 
    the leawst cost Jcv.
    Now, to evaluate the performacne of your model, you can use the training set to check the 
    performance which will give a fair estimate of your model as this testing data is completely
    new to the model and is not optimistic (overfitting) to say training data / CV data.

    CV is also called as validation set / development set/ Dev set

    So with this procedure u might be able to chose a model that will best fit your data.


3. Get to know about the bias and variance of the model.
    Underfit - High bias - High Jtrain,Jcv
    Overfit - High variance - Low Jtrain,High Jcv

    Once you get to know whether your model is having high bias/ high variance , you can
    apply ur metrics to improve its performance

    In traditional ML(Linear/Logistic regression):
    - To solve High bias
        - Increase more features
        - Increase the complexity of NN
        - Try adding polynomial 
        - decreasing Lambda (regularisation parameter)

    - To solve High variance
        - Get more training data like data augmentation / data synthesis etc
        - try reducing features / complexity in NN
        - increasing lambda(regularisation parameter)
        - Try decresing the training epochs using Earlystopping()
        - dropout

    In neural networks:
    - Larger neural networks almost always are low bias machines(learns perfectly on training data)
    - In general terms , If model has high bias -> increse the neural network size.
      & if model has high variance -> get more data.
    - Sometimes its difficult to increase the neural network size (already too large) or get more
      data & it depends to optimise such models
    - Usually in neural networks , increasing the size doesnt create high variance if you adjusted
      regularisation parameter, larger one is almost always equal or better than the smaller neural
      network.


4. Error analysis can also be a very great tool for you to understand on how to imrpove 
   the performace of the machine larning model.
   - Say, you have 500 CV examples & 100 of them are misqualified
   - You can actually manually check on these examples and try to analyse, which kind of examples
     are miscassified by the algorithm and try to get more data of such kind or add new features 
     which could capture such patterns etc.


     
Adding more Data : 
1. If it is difficult to add more data of everything , try to add more data of the type error
   analysis has indicated it might help.

2. Data augmentation: Modifying the existing data to create more data.
   (Works in audio or text recognition very well, like adding noise to the voice/text and generating
   more training data.)

3. Data synthesis: Creating your own data from scratch
   (In some cases it is possible, so u can think of this option, Mostly used in CV tasks)

4.Sometimes Transfer laerning can help you to save your time & also properly train your model,
  even with less data because it has already been trained on much larger set of data.
  (Use the pre trained models & then fine tune them according to your needs).

5.


    
So in the development of machinea learning model:
1. First chose architecture (Model, data..etc)
2. Train the model
3. Run the diagnostics (Bias , Variance & error analysis)
4. Make necessary changes and go the step 1.
    
      

'''