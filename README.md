# Investigating the transferability of Reinforcement learning-based Hyperparameter optimisation methods
We provide here the source code for the paper "Investigating the transferability of Reinforcement learning-based
Hyperparameter optimisation methods".

## Environment
To generate a virtual environment within which you can run the code, look at the requirement.yml file.

## Code organisation
You can find the code for both agents in the agent folder, and the weights for them in Users/janrichtr/... and UsersPPO/janrichtr/... directories.

The data that is used for training the agent is located in metadata/nn-meta/split-0, where the data for training environment is in a separate file.

To train the DQN agent, run the run_nn_meta.py script. Similarly, to train the PPO agent run the run_nn_meta2.py script.

The custom tabular reinforcement learning environments, that were used to evaluate the HPO methods are in complexmaze.py and simplemaze.py files.

To evaluate the traditional HPO baselines, use the bo.py and random_search.py files.

To evaluate the agents, run the eval.py and eval_ppo.py script for the DQN and PPO agent respectively.

## Results folder

All of the data that was used to generate the figures can be found in the .txt files, the files are labelled according to the method and the number of trials

To extract the data from these files use the python scripts containing word "extraction" in them. Copy the contents of the .txt file you want to extract. It returns an to then use for the generation of graphs.

To generate the figures as can be found in the paper use the rq1.py, rq2.py and rq3.py scripts. They will generate the graphs for the corresponding research sub-question.


