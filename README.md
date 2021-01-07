## Training data
Training sets come from [HGMD](http://www.hgmd.cf.ac.uk/)
Details can be found in the Supplementary Table2

## Annotation
Some of features were annotated by the [ANNOVAR](https://doc-openbio.readthedocs.io/projects/annovar/en/latest/).  
- A) AFs and GFs of variants estimated from 125,748 exomes in [Genome Aggregation Database (gnomAD)](https://gnomad.broadinstitute.org) (version 2.1.1); 
- B) Conservation information, i.e., [CCRs](https://s3.us-east-2.amazonaws.com/ccrs/ccr.html)，phastCons;
- C) Secondary structure information, i.e., Beta strand, Helix, Turn, Coll.
- D) Domain information, i.e., VWFC domain, Fibrillar Collagen NC1 domain.
- E) Region information, i.e., N-terminal region, Triple-helical region, C-terminal region, Collagen alpha-2(I) chain, Signal peptide
- F) Modification information, i.e., Disulfide bond
- G) Motif information, i.e., Cell Attachment Site motif
- H) Variant Type, i.e., del<10bp, del>10bp, Gly-Ala, Gly-Arg, Gly-Asp, Gly-Cys, Gly-Glu, Gly-Ser, Gly-Val, Gly-Trp, ins<10bp, ins>10bp, non GLY, nonsense
- I) Gene function information, i.e., exonic, splicing, intronic
- J) Mutation Type of variants, i.e., frameshift, non-frameshift, nonsynonymous, start-loss, stop-gain, synonymous

## Training
The predition model was trained using the python package [LightGBM](https://github.com/microsoft/LightGBM) (version 2.3.1), and 
parameters were tuned by [Bayesian optimization](https://github.com/fmfn/BayesianOptimization). The random status was 
set as `1` throughout the model training process.

## Environments
The environments of mvPPT built in our study: 
- python 3.7.4
- sklearn 0.22.1
- pandas 0.25.3
- lightGBM 2.3.1
- bayesian-optimization 1.1.0
