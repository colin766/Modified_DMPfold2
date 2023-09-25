# Modified_DMPfold2
In network.py, the self-attention, coordinate attention and ConvNextV2 stage are implemented, where the self-attention is not used. the default value of these implementation is False, which means if you do not turn it on, these will not appear in the model architecture. 

In line 385 and line 399, code can control whether using coordinate attention and ConvNextV2, if set the use_self_attention and use_convnext_v2_block to true, the coordinate attention and convNextV2 will be active. 

code citation 


	This project used the DMPfold2 as the basic architecture model to modify.
The GitHub link is: https://github.com/psipred/DMPfold2

	The Self-attention layer are used as implementation layer.
The GitHub link is: https://github.com/heykeetae/Self-Attention-GAN.

	The Coordinate attention layer are used as implementation layer.
The GitHub link is: https://github.com/houqb/CoordAttention

	The ConvNextV2 are used to instead the stage in Maxout2D layer.
The GitHub link is: https://github.com/facebookresearch/ConvNeXt-V2

the training data download from:
https://rdr.ucl.ac.uk/articles/dataset/Protein_structures_predicted_using_DMPfold2/14979990
