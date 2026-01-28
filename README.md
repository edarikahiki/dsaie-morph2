# Predictive River Morphology with Better JamUNet
**by group MORPH 2**

Submitted for Data Science and Artificial Inteligence for Engineers (CEGM 2003) Course\
Faculty of Civil Engineering & Geoscience, Delft University of Technology\
Final presentation slides can be found at [Final Presentaion](https://www.example.com)

---
## Project Overview

The project is built based on original work by Antonio Magherini ([Github repo link](https://github.com/antoniomagherini/jamunet-morpho-braided.git)), which tries to make a predictive morphological changes of River Brahmaputra-Jamuna using satellite images input with JamUNet model. Our project goal is to improve the original model by modifying architecture, tuning hyperparameter, trying different input output combination, and using higher input image resolution. We provide 20 different models that experimenting on various settings. The result indicating model improvement on some scenarios. However, some other results do not perform better than the original model. We hope that our work could be useful as recommendation for future similar projects.

---
## Repository Structure
This repository is structured as follows:
- `miscellaneous` : Consist of notes on weekly progress and other miscellaneous
- `model` : Consist of notebook files to run models and model results
- `Postprocessing` : Consist of modules for data postprocessing
- `Preprocessing` : Consist of modules for data preprocessing    

---
## Models Explanation

**Architecture tests (batch size = 16, initial hidden layer = 8)**

- <code>00. Original.ipynb</code>: Baseline JamUNet model that predicts the next year water mask for a 4-year input stack. This serves as the reference for all comparisons. 

- <code>01. 25D.ipynb -> NEW NAME 3D</code>: Uses a 3D temporal convolution only in the first block then switches to standard 2D convolutions for the rest of the U-Net

- <code>02. Semi3D.ipynb</code>: Applies a 3D convolution with temporal kernel dependend on input temporal size dimension in the first block and for the rest of the block perfrom 2D convolution using 3D function with temporal kernel of 1

- <code>03. Bottleneck3D.ipynb</code>: Applies a 3D temporal convolution only at the bottleneck layer

- <code>04. AddLayer.ipynb</code>: Adds one additional layer on top of the best-performing architecture 3D 

**Hyperparameter tests (Original architecture, 60m resolution dataset)**

- <code>05. Batch4.ipynb</code>: original model but trained with batch size 4.

- <code>06. Batch32.ipynb</code>: original model but trained with batch size 32. 

- <code>07. HidLayer32.ipynb</code>: original model but with 32 initial hidden layers. 

- <code>08. HidLayer64.ipynb</code>: original model but with 64 initial hidden layers.  

**Input-Output combinations (60 m resolution dataset with best architecture (3D) and original hyperparameters)**

- <code>09. 5to1.ipynb</code>: Predicts one target year using five preceding yearly inputs. 

- <code>10. 3to1.ipynb</code>: Predicts one target year using three preceding yearly inputs. 

- <code>11. 2to1.ipynb</code>: Predicts one target year using two preceding yearly inputs. 

- <code>12. 1to1.ipynb</code>: Predicts one target year using only the previous year. 

**Higher image resolution (30 m resolution dataset with best architecture (3D) and original hyperparameters)**

- <code>13. 30full.ipynb</code>: Trains and evaluates the best architecture on the full 30 m images (2000×1000), increasing spatial detail relative to 60 m.

- <code>14. 30fullAddLayer.ipynb</code>: Adds one additional layer on top of <code>13. 30full.ipynb</code> model. 

- <code>15. 30full_split.ipynb</code>: Trains on 30 m images but split it into two tiles (1000 x 1000). 

- <code>16. 30full_32hid.ipynb</code>: Same as the <code>13. 30full.ipynb</code> model but with 32 initial hidden layers.

**Hyperparameter tested on best architecture 3D (60 m resolution dataset)**

- <code>17. Batch4_25D.ipynb</code>: Best-architecture model (3D) trained with batch size 4. 
- <code>18. Batch32_25D.ipynb</code>: Best-architecture model (3D) trained with batch size 32. 
- <code>19. HidLayer32_25D.ipynb</code>: Best-architecture model (3D) trained with 32 initial hidden layers.
- <code>20. 30full_split_samevalidation.ipynb</code>: Split-tile 30 m training (1000×1000) while keeping validation on the full 2000×1000 shape. 

---
### Extra Notes
- To run the model with dependencies safely, make sure miniconda & pytorch are installed
- Use `braided_mod.yml` as environment file
- Guidance to install conda and environment dependencies can be found in `setup.sh`
- Due to github storage limitation, input image data (both 60m and 30m resolution) and saved best model are not included 
---

### References

Magherini, A. (2024). *JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning* (Master’s thesis, Delft University of Technology).  
https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348

Jean-Francois Pekel, Andrew Cottam, Noel Gorelick, Alan S. Belward, High-resolution mapping of global surface water and its long-term changes. Nature 540, 418-422 (2016). (doi:10.1038/nature20584)

<details>
<summary>BibTeX</summary>

```bibtex
@mastersthesis{magherini2024,
  author       = {Magherini, A.},
  title        = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
  school       = {{Delft University of Technology}},
  year         = {2024},
  month        = {10},
  howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
