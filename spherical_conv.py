'''Collection of Keras Layers to create SFNO Keras Model'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import TensorDynamics.sphere_harm_new as sh
from tensorflow import keras

#############################################################
# DUE TO TensorDynamics, ONLY BATCH SIZE OF ONE IS SUPPORTED
#############################################################

# set which activation type to use
act_type="gelu"

class SpherePosEmb(keras.layers.Layer):
    """ Python object that performs positional embedding, with identical embeddings along the longitude coordinate

    """
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        """
        Given input shape (batch, level, latitude, longitude), initialize weights for positional embedding
        """
        # grab dimension lengths for level and latitude
        shape=(input_shape[-3],input_shape[-2],1)

        # initialize embedding weights with random values
        init_vals=(tf.random.normal(shape=shape,dtype=tf.float32))
        self.w=tf.Variable(init_vals,trainable=True,dtype=tf.float32)

    def call(self, inputs):
        """
        Given input data, add positional embeddings
        """
        return inputs +self.w

#############################################################################################################################

class SphereConv(keras.layers.Layer):

    """ Python object that performs convolutions in spherical harmonic space. Interpolate filter weights between wavenumbers to 
        enforce locality

    """
    def __init__(self, filters=8,relu=False):
        super().__init__()
        self.filters = filters # number of filters to use
        self.relu=relu # whether to use relu on the real part of the spherical harmonic coefficients

    def build(self, input_shape):
        """ Create the weights for the spherical harmonic convolutions, separately for the real and imaginary parts

        """

        # array containing the total wavenumbers
        self.l = tf.cast(tf.math.sqrt(1+tf.range(0,input_shape[-2],dtype=tf.float32))[:,None],np.csingle)

        # Only set weights for every 4th wavenumber
        interval=4
        N=tf.cast(input_shape[-2]/interval,tf.int32)
        shape=(self.filters,input_shape[-3],N,1)

        # initiate weight values for both the real and imaginary parts
        real_vals=tf.random.normal(shape=shape,dtype=tf.float32)
        imag_vals=tf.random.normal(shape=shape,dtype=tf.float32)
        self.w_real=tf.Variable(real_vals,trainable=True)
        self.w_imag=tf.Variable(imag_vals,trainable=True)


        # helper arrays/values to perform the interpolation of the weights
        self.x_targ=tf.cast(tf.range(0,input_shape[-2]),dtype=tf.float32)
        self.x_ref_min=tf.constant(0,dtype=tf.float32)
        self.x_ref_max=tf.constant(input_shape[-2]-1,dtype=tf.float32)

    def call(self, inputs):
        """ Perform the convolution in spherical harmonic space

        """

        # interpolate to get the weight values at each wavenumber
        reals=tfp.math.interp_regular_1d_grid(self.x_targ,self.x_ref_min,self.x_ref_max,self.w_real,axis=-2)
        imags=tfp.math.interp_regular_1d_grid(self.x_targ,self.x_ref_min,self.x_ref_max,self.w_imag,axis=-2)

        # combine the real and imaginary parts of the weights, then perform the convolution
        weights=tf.dtypes.complex(reals, imags)
        x=tf.math.reduce_mean(weights*inputs,axis=-3)*self.l

        # perform relu if specified
        if self.relu:
            rx=tf.math.real(x)
            ix=tf.math.imag(x)
            rx=tf.nn.relu(rx)
            x=tf.dtypes.complex(rx, ix)
        return x
    
#############################################################################################################################

class MLP(keras.layers.Layer):
    """ Multilayer perceptron block with one hidden layer, applied pointwise

    """
    def __init__(self, nlevels,hlevels,activation=None,doNorm=False):
        """ Create the block, consisting of two MLP layers, and one normalization layer

        """
        super().__init__()

        # hidden layer
        self.MLP1=keras.layers.Conv2D(hlevels,1,activation=act_type)

        # output layer
        self.MLP2=keras.layers.Conv2D(nlevels,1,activation=activation)

        # normalization layer
        self.norm=keras.layers.GroupNormalization(groups=-1,center=True)

        # whether to apply the normalization
        self.doNorm=doNorm

    def call(self, inputs):
        """ Perform the dense transformations, then apply normalization if desired

        """
        x = self.MLP2(self.MLP1(inputs))
        
        if self.doNorm:
            return self.norm(x)
        else:
            return x

#############################################################################################################################

    
class SFNO_block(keras.layers.Layer):
    def __init__(self, grid_obj,nlats,nlevels):
        """ Create the SFNO block, initializing the requisite MLP and spherical convolution blocks

        """
        super().__init__()

        # initialize grid parameters and spherical harmonic object
        self.grid_obj=grid_obj
        self.nlats=nlats
        self.nlevels=nlevels

        # store layers for the spherical convolution
        self.sphere_conv=SphereConv(filters=nlevels,relu=False)

        # store layers for the two MLP blocks
        self.MLP1=MLP(nlevels,nlevels,activation=act_type)
        self.MLP2=MLP(nlevels,nlevels,activation=act_type,doNorm=True)

    def call(self, inputs):
        """ perform the SFNO block operations

        """

        # get spherical harmonic coefficients, then perform spherical convolution 
        harms=self.grid_obj.calc_sh_coeffs(inputs)
        harms=self.sphere_conv(harms)

        # convert back to grid space, perform gelu activation
        grids=self.grid_obj.eval(harms,self.grid_obj.legfuncs) 
        grids=tf.nn.gelu(grids)

        # transpose and add to first MLP output
        grids=tf.transpose(grids[None,...],perm=[0,3,2,1])
        x1=tf.transpose(inputs,perm=[2,1,0])
        x1 = self.MLP1(x1[None,...])
        added= x1+grids

        # perform second MLP, transpose back to original shape
        output=self.MLP2(added)
        output=tf.transpose(output,perm=[0,3,2,1])

        # add residual and return
        return output[0]+inputs



#############################################################################################################################

    
class SphereModel(keras.Model):
    """ Python object contains spherical harmonic grid object and performs encoding, decoding, and SFNO blocks

    """
    def __init__(self,nlats=32,nlevels=32,embed_dim=64,nblocks=4,dfact=2,loops=1):
        super().__init__()

        # grid parameters and spectral truncation
        self.nlats=nlats
        self.nlevels=nlevels
        self.trunc=tf.cast(nlats/dfact,tf.int32)

        
        # spherical harmonic objects for the full and the truncated expansions
        self.grid_obj_up = sh.sh_obj(nlats=nlats,trunc=nlats)
        self.grid_obj_down = sh.sh_obj(nlats=self.trunc,trunc=self.trunc)

        # initiate encoder MLP and positional embedding
        self.encoder=MLP(embed_dim,embed_dim,activation=None,doNorm=False)
        self.pos_embed=SpherePosEmb()

        # create a list with the desired number of SFNO blocks
        self.layer_list=[]
        for i in range(nblocks):
            self.layer_list.append(SFNO_block(self.grid_obj_down,self.nlats,embed_dim))

        # create decoder MLP
        self.decoder=MLP(nlevels,embed_dim,doNorm=False)
        
    def call(self, inputs):

        # transpose to needed shape (batch, level, lat, lon)
        x=tf.transpose(inputs,perm=[0,3,2,1])

        # encode to the embedding dimension
        res=self.encoder(x)

        # transpose to (batch, lat, lon, embedding), then add positional embedding
        x=tf.transpose(res,perm=[0,3,2,1])
        x=self.pos_embed(x)

        # convert to the truncated resolution
        x=self.grid_obj_up.calc_sh_coeffs(x[0])
        x=self.grid_obj_down.eval(x[:,:self.trunc+1,:self.trunc+1],self.grid_obj_down.legfuncs)

        # perform the SFNO blocks
        for layer in self.layer_list:
            x=layer(x)

        # upscale back to the original resolution
        x=self.grid_obj_down.calc_sh_coeffs(x)
        x=tf.pad(x,[[0,0],[0,self.nlats-self.trunc],[0,self.nlats-self.trunc]])
        x=self.grid_obj_up.eval(x,self.grid_obj_up.legfuncs)

        # transpose and perform decoder MLP
        x=tf.transpose(x,perm=[2,1,0])[None,...]+res
        x=self.decoder(x)

        # transpose back to original shape
        x=tf.transpose(x,perm=[0,3,2,1])

        return x
