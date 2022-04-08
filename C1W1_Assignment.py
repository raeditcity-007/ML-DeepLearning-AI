#!/usr/bin/env python
# coding: utf-8

# # Week 1 Assignment: Housing Prices
# 
# In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
# 
# Imagine that house pricing is as easy as:
# 
# A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.
# 
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
# 
# Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

# In[15]:


import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


# In[16]:


# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE
    
    #List with possible number of bedrooms (up to 6)
    number_of_bedrooms = [1,2,3,4,5,6]
    
    #start an empty list for the costs of the houses
    house_cost = []
    
    #fill the list with the house costs
    for i in number_of_bedrooms:
        house_cost.append(50 + i*50)
        #costs in thousands of dollars
    
    #convert the lists to NumPy arrays
    #guarantee that the arrays store float variables
    number_of_bedrooms_array = np.array(number_of_bedrooms, dtype = float)
    house_cost_array = np.array(house_cost, dtype = float)
    
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    
    #Reshape the ys for the correct shape
    xs = number_of_bedrooms_array
    ys = house_cost_array.reshape(-1,1)
    print("xs array:")
    print(xs)
    print("ys array:")
    print(ys)
    
    #Normalize y's dividing by 100
    ys = ys/100
    
    #Modelling with the ANN
    
    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = Sequential()
    model.add(Dense(units=1, input_shape=[1]))
    
    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs = 1000)
    
    ### END CODE HERE
    return model


# Now that you have a function that returns a compiled and trained model when invoked, use it to get the model to predict the price of houses: 

# In[17]:


# Get your trained model
model = house_model()


# Now that your model has finished training it is time to test it out! You can do so by running the next cell.

# In[18]:


new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)


# If everything went as expected you should see a prediction value very close to 4. **If not, try adjusting your code before submitting the assignment.** Notice that you can play around with the value of `new_y` to get different predictions. In general you should see that the network was able to learn the linear relationship between `x` and `y`, so if you use a value of 8.0 you should get a prediction close to 4.5 and so on.

# **Congratulations on finishing this week's assignment!**
# 
# You have successfully coded a neural network that learned the linear relationship between two variables. Nice job!
# 
# **Keep it up!**
