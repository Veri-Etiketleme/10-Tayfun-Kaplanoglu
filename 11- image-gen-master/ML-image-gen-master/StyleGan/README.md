**StyleGAN2 Distillation for Feed-forward Image Manipulation**

StyleGAN2 is a generative model made by Nvidia. This model possesses certain feature morphing capabilities via its latent variable mapping architecture such gender shift and varying the extents of attributes. Standalone, its only function is generation of images that do not exist.
This project makes use of StyleGAN2-ffhq-config-e which generates facial data.

The aim of this project is to distill these capabilities into a pipeline model that is trained on latent variables that encompass these attributes.
The mathematics used is based on a combination of Evgeny Kashins research paper and the linear algebra manipulation I derived that was specific to my desired outputs.

The steps include manipulating and using the SttleGAN2 repository by NVLabs to suit data generation and manipulation.

Then extracting the intermediate latent variables( and choosing the right ones) and proceeding to shift characteristics based on certain mathematical equations

This would lead to a pair dataset generation that visually encompasses the shift in its latent code.

This pair is trained through a pipeline model that accepts two sets A- the non modified visual data and B- the modified equivalents and trains the pipeline to perform a similar manipulation.

The outputs of this pipeline train weren't perfect even after 12 hours of continuous training after downscaling to 256p. But as the transfer and model improvement were significantly apparent, the training was concluded.

New images fed to this model now outputted a more feminine version of the person. This was tested on the celeb-A dataset by tensorflow

Two sub projects I performed were:

- Image morphing( Smooth shift from one face to another and viewing the intermediate people between them)
- Expression transfer( Variation of smiling and frowning although it could be improved with a purer latent distillation and inclusion of features like eye concavity, nose flare and brow arch)
