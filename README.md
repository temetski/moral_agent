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