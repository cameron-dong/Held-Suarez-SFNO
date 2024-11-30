import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time


def get_train_step(model,optimizer,L1_loss,spec_loss,spec_ratio,loops):
    """
    Args:
        model - keras model object
        optimizer - keras optimizer
        L1_loss - MAE loss function
        spec_loss - spectral energy loss function
        spec_ratio - scaling for spec_loss
        loops - number of iterations for model
    Returns:
        train_step - function to perform a single training step
    """
    
    @tf.function(jit_compile=True,reduce_retracing=True)
    def train_step(x, y):
        """
        Performs training step, updating model weights
           
        Args:
            x - initial data
            y - target data
        Returns:
            loss value normalized by number of loops
        """

        # perform backpropagation
        with tf.GradientTape() as tape:

            # perform single loop, save output
            output = model(x)
            MS_pred=tf.identity(output) # holder for mean of output values ("mean state")

            # loss for first loop
            loss_value = L1_loss(y[0], output)+spec_ratio*spec_loss(y[0],output)

            # perform desired number of loops, adding loss for each
            for i in range(loops-1):
                output = model(output)
                MS_pred=MS_pred+output
                loss_value = loss_value+L1_loss(y[i+1], output)+spec_ratio*spec_loss(y[i+1],output)

            # calculate loss for prediction averaged over all loops, then add to total loss
            MS_pred=MS_pred/loops
            MS_true=tf.math.reduce_mean(y[0:loops],axis=0)
            loss_value=loss_value+0.2*loops*L1_loss(MS_true,MS_pred)

        # update model weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value/loops

    return train_step

#############################################################################################################
#############################################################################################################

def get_val_step(model,L1_loss,spec_loss,loops):
    """
    Args:
        model - keras model object
        L1_loss - MAE loss function
        spec_loss - spectral energy loss function
        loops - number of iterations for model
    Returns:
        val_step - function to calculate losses for a single validation step
    """
    
    @tf.function(jit_compile=True)
    def val_step(x, y):
        """
        Args:
            x - initial data
            y - target data
        Returns:
            float: loss value normalized by number of loops
        """

        # perform initial iteration and loss calculation
        output = model(x)
        L1_value = L1_loss(y[0], output)
        spec_val = spec_loss(y[0],output)

        # perform rest of loops, adding to the loss
        for i in range(loops-1):
            output = model(output)
    
            L1_value = L1_value+(L1_loss(y[i+1], output))
            spec_val = spec_val+spec_loss(y[i+1],output)
    
        # return the per loop loss
        return [L1_value/loops,spec_val/loops]
    return val_step

#############################################################################################################
#############################################################################################################

def training_loop(values,train_step,val_step,epochs,loops,n_train,n_vali):
    """
    Trains the keras model used in train_step function
    
    Args:
        values - tensor of training and validation data
        train_step - training step function
        val_step - validation step function
        epochs - number of epochs to do
        loops - rollout length
        n_train - number of training samples
        n_vali - number of validation samples
    """

        # loop over the number of epochs
    for epoch in range(epochs):
        etime=time.time()
        print("\nStart of epoch %d" % (epoch+1,))

        # create empty lists for tracking runtime and training loss
        runtimes=[]
        losses=[]

        # shuffle order that we progress through the training data
        order = np.arange(n_train)
        np.random.shuffle(order)

        # iterate over training data
        for i,step in enumerate(order):
            stime=time.time()

            # perform training step
            loss_value=train_step(values[step:step+1],values[step+1:step+loops+1])

            # record runtime and loss
            runtimes.append(time.time()-stime)
            losses.append(loss_value)

            # print to screen the progress and expected remaining runtime
            if i % 10 == 0:
                print(" %s/%s samples, " % ((i + 1),n_train),"eta: %ss" %
                    (np.round((n_train-i)*np.median(runtimes),1)),"loss: %s" % (np.half(np.mean(losses))),end='\r')
                    
        # create empty list to track validation loss, then iterate over validation data
        val_losses=[]
        for step in range(n_train,n_train+n_vali-loops):
            val_losses.append(val_step(values[step:step+1],values[step+1:step+loops+1]))

        # print validation loss
        val_loss=np.half(np.mean(val_losses,axis=0))
        print(" %s/%s samples, time taken: " % ((step + 1),n_train),np.round((time.time()-etime),1),
            "  loss: %s" % (np.half(np.mean(losses))),"  val loss:", (val_loss),end='\n')


