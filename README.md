

Install Anaconda and create the virtual environment using the environment file: conda env create -f env.yml

To Run the SQIA Experiments, execute the following command:
python GS_Paper-All_Experiments.py -pid <5/6/7/72> -bp '<result_location>' --no-noisy -r '<repeat the experiment>' -sp '<delta>'

You can also run without the repeat and success probability parameter. They will take the default values
python GS_Paper-All_Experiments.py -pid <5/6/7/72> -bp '<result_location>' --no-noisy

To Run the OQIA Experiments, execute the following command:
python QAOA_Paper-All_Experiments.py -pid <5/6/7/71> -bp '<result_location>' --no-noisy --inc -r '<repeat the experiment>' -p '<repetation depth of QAOA>'

You can also run without the repeat and success probability parameter. They will take the default values
python QAOA_Paper-All_Experiments.py -pid <5/6/7/71> -bp '<result_location>' --no-noisy --inc

To Generate the table after all the executions
python gen_table.py -bp '<result_location>' -r '<repeat the experiment>'

Again repeat is option. It will take the default value of 10
