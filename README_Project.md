# Ethics-in-the-Loop RL
### Authors
Julia Santaniello
Joey Grossman
Angel Shen
Matt
Manik

### Overview
This repository includes code for collecting demonstrations, training an agent using Q-Priming, GAIL and running computational experiments. Plotting is available in the AA_graphs.ipynb Jupyter Notebook.

### To Run
#### Demonstrations
To Collect Demonstrations:
```
python socket_agent_demonstration.py
```

To View Demonstrations:
```
python socket_agent_performing_demo.py
```

#### Q Priming
To Train with QPriming:
```
python socket_agent_training_QPrime.py
```

To View Agent Trained with QPriming:
```
python socket_agent_performing_QPrime.py
```

#### GAIL
To Train with GAIL:
```
python socket_agent_training_GAIL.py
```

To View Stitched Policies Trained with GAIL:
```
python socket_agent_performing_GAIL.py
```

#### Running Experiments

```
python run_pipeline.py
```

Then run all cells in 
```
AA_graphs.ipynb
```

There is a propper shopper bug that freezes learning if the agent leaves with a cart. In that case, please rerun the experiment.

