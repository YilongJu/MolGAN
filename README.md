# LOMolGAN
 Modification of original MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)

 ## Dependencies

 * **python>=3.7**
 * **tensorflow>=1.7.0**: https://tensorflow.org
 * **rdkit**: https://www.rdkit.org
 * **numpy**
 * **scikit-learn**

 ## Usage
 - Modify hyperparameters in the `main` function of `example.py`
 - To train:
    - Set `skip_training` to `True`
 - To test:
    - Keep hyperparameters unchanged
    - Set `skip_training` to `False`
    - Set `test_epoch` to the epoch used for inference
 
 Then, type
 ```
 python example.py
 ```
 to run.

 ## Check results
 Figures of generated molecules are saved as a **png** file in a folder whose name begins with **qm9_5k...**, which is records the experiment setting.
