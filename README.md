# PoET: A generative model of protein families as sequences-of-sequences

This repo contains inference code for ["PoET: A generative model of protein families as sequences-of-sequences"](https://arxiv.org/abs/2306.06156), a state-of-the-art protein language model for variant effect prediction and conditional sequence generation.

## Environment Setup

1. Have `mamba` (faster alternative to `conda`) installed ([Instructions](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html))
1. Have `conda-lock` installed in your base conda/mamba environment ([Instructions](https://github.com/conda/conda-lock#installation))
1. Run `make create_conda_env`. This will create a conda environment named `poet`.
1. Run `make download_model` to download the model (~400MB). The model will be located at `data/poet.ckpt`. Please note the [license](#License).

## Scoring variants

Use the script `scripts/score.py` to obtain fitness scores for a list of protein variants given a MSA of homologs of the WT sequence.

1. Be on a machine with a NVIDIA GPU. The model cannot run on CPU only.
1. Activate the `poet` conda environment
1. Run the script, replacing the values in angle brackets with the appropriate paths.

   ```
   python scripts/score.py \
   --msa_a3m_path <path to MSA of homologs of WT sequence> \
   --variants_fasta_path <path to fasta file containing variants to score> \
   --output_npy_path <path to output file where scores for each variant will be stored as a numpy array>
   ```

You can pass a lower value for the batch size (`--batch_size`) if you run out of VRAM. The script was tested on an A100 GPU with 40GB VRAM.

## Example

Run the scoring script without arguments `python scripts/score.py` to score variants in the `BLAT_ECOLX_Jacquier_2013` dataset from ProteinGym.

- the dataset is located at `data/BLAT_ECOLX_Jacquier_2013.csv`
- the variants to score as a fasta file is located at `data/BLAT_ECOLX_Jacquier_2013_variants.fasta`
- the MSA of homologs of the WT sequence, generated using ColabFold MMseqs2 with the UniRef2202 database, is located at `data/BLAT_ECOLX_ColabFold_2202.a3m`
- the scores will be saved as a numpy array at `data/BLAT_ECOLX_Jacquier_2013_variants.npy`

The scores obtained from the script should obtain `>0.65` Spearman correlation with the measured fitness (DMS_score column in the dataset file).

## Citation

You may cite the paper as

```
@inproceedings{
      poet_neurips2023,
      title={PoET: A generative model of protein families as sequences-of-sequences},
      author={Timothy F. Truong Jr and Tristan Bepler},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      url={https://openreview.net/forum?id=1CJ8D7P8RZ}
      volume = {37},
      year={2023},
}
```

## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

The [PoET model weights](https://zenodo.org/records/10061322) (DOI: `10.5281/zenodo.10061322`) are available under the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license for academic use only. The license can also be found in the LICENSE file provided with the model weights. For commercial use, please reach out to us at contact@ne47.bio about licensing. Copyright (c) NE47 Bio, Inc. All Rights Reserved.
