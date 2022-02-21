This project attempts Neural Style Transfer with the new constraint of sticking to ResNet Architecture.

NST is widely taken up but a majority of the good results are obtained by sticking to VGG or Tensorflows NST model. 

In this project I have analyzed different layers of ResNet architecture to find the optimal content and style layers for NST as well as implemented a the content and style losses for NST.

The main challenge was to obtain a robust feature transfer as ResNet is not by default an optimal choice of model for NST as its feature capture is not as robust as other nets.


![Robustness Graph](<./robustness_graph.jpeg>) 

Results with trials can be viewed in NST__trials.pptx in this repo
