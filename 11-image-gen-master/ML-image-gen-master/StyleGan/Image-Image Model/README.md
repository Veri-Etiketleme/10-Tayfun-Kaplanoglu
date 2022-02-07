This is supposedly the most advanced mode of distillation I attempt in this project set.

AIM: To train an image-image pipeline model on a pair dataset generated using A(StyleGAN Generated Data) and B(A modified in latent space as seen in expression or morphing distillation.)

This pipeline model once trained must be able to perform the equivalent latent operations on normal human pictures.

The pair B has been taken as the gender modified dataset( latent code + z, where z represents the latent vector for gender(more femenine))

The model weights are stored in latest_net_G.pth

