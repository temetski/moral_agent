## Setup
Create and activate a virtual env [optional]
```
python -m venv venv
source venv/bin/activate
```
And then install all prerequisite packages
```
pip install -r requirments.txt
```


## Pretraining a model

Models can be pretrained using the following command:
```
python algorithms/ppo.py --seed 42 --env-id environments.milk:FindMilk-v4 --num-envs 8
```

Note that the `--env-id` parameter uses the format <path.to.module>:<name_of_env>. 


## Citing
If your find any part of this work useful to your research, your are encouraged to cite our paper:

```bib
@article{amuled2026,
  author = {Dubey, Rohit K. and Dailisan, Damian and Mahajan, Sachit},
  title = {Addressing Moral Uncertainty using Large Language Models for Ethical Decision-Making},
  journal = {Frontiers in Artificial Intelligence},
  year = {2026},
  doi = {10.48550/ARXIV.2503.05724},
  note = {Accepted for publication},
  keywords = {Computers and Society (cs.CY), Artificial Intelligence (cs.AI)},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
