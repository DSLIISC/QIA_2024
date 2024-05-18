This repository contains the source code to run the experiments in the "Index Advisors on Quantum Platforms" paper. 

Install Anaconda and create the virtual environment using the environment file: conda env create -f env.yml

To Run the SQIA Experiments, execute the following command:
python GS_Paper-All_Experiments.py -pid <5/6/7/72> -bp '<result_location>' --no-noisy -r <repeat the experiment> -sp <delta>

To Run the OQIA Experiments, execute the following command:
python QAOA_Paper-All_Experiments.py -pid <5/6/7/71> -bp '<result_location>' --no-noisy --inc -r <repeat the experiment> -p <repetation depth of QAOA>