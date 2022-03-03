Self-Supervised Vision Transformers Learn Visual Concepts in Histopathology
===========
<details>
<summary>
  <b>Self-Supervised Vision Transformers Learn Visual Concepts in Histopathology</b>, LMRL Workshop, NeurIPS 2021.
  <a href="https://arxiv.org/abs/2203.00585" target="blank">[PDF]</a>
  <br><em>Richard. J. Chen, Rahul G. Krishnan</em></br>
</summary>

```bash
@article{chen2022self,
  title={Self-Supervised Vision Transformers Learn Visual Concepts in Histopathology},
  author={Chen, Richard J and Krishnan, Rahul G},
  journal={Learning Meaningful Representations of Life, NeurIPS 2021},
  year={2021}
}
```
</details>

<div align="center">
  <img width="80%" alt="DINO illustration" src=".github/Pathology_DINO.jpg">
</div>

## Pretrained Models
<table>
  <tr>
    <th>Arch</th>
    <th>SSL Method</th>
    <th>Epochs</th>
    <th>Dim</th>
    <th>CRC100K-R</th>
    <th>CRC100K-N</th>
    <th>Download</th>
  </tr>
  
  <tr>
    <td>ResNet-50 (Trunc)</td>
    <td>ImageNet Transfer</td>
    <td>N/A</td>
    <td>1024</td>
    <td>0.935</td>
    <td>0.983</td>
    <td>N/A</td>
  </tr>
  
  <tr>
    <td>ResNet-50</td>
    <td><a href="https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py">SimCLR</a></td>
    <td>100</td>
    <td>2048</td>
    <td>0.938</td>
    <td>0.981</td>
    <td><a href="https://github.com/Richarizardd/Self-Supervised-ViT-Path/blob/master/ckpts/resnet50_tcga_brca_simclr.pt">Backbone</a></td>
  </tr>
  
  <tr>
    <td>ViT-S/16</td>
    <td><a href="https://github.com/facebookresearch/dino">DINO</a></td>
    <td>100</td>
    <td>384</td>
    <td>0.941</td>
    <td>0.987</td>
    <td><a href="https://github.com/Richarizardd/Self-Supervised-ViT-Path/blob/master/ckpts/vits_tcga_brca_dino.pt">Backbone</a></td>
  </tr>
</table>
