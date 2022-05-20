# plan-CVAE

This is the PyTorch implementation for inference and training of the plan-CVAE framework as described in:

> **Exploring Latent Dynamics for Visual Storytelling**

Inspired by the visual story creation process that humans always consider the influence of the next sentence before they continue the story writing, we address the story generation by learning an information dynamic plan in a latent space for every sentence.

<p align="center">
     <img src="https://github.com/feizc/Latent-Dynamics/blob/main/images/case.png" alt="Latent Dynamics">
     <br/>
     <sub><em>
      Generating visual story conditioned on latent dynamics. <br/> 
      A latent variable is first generated with latent dynamic module according to history story. A visual story generator then conditionally generates the next coherent sentence with this latent plan guiding.
    </em></sub>
</p>


## Model Structure 

CVAE framework with latent variables modeling for sentence-level consistency

## Dataset 

[VIST](https://visionandlanguage.net/VIST/) 

