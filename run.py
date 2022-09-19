from global_settings import PATH_DICT
from global_settings import get_dir
from params.params import num_trials
from main import plotting
from main import experiment
from main import experiments
from torch import nn
import os


if __name__ == "__main__":
    model_name = "mleauto"
    exp_path = PATH_DICT[model_name]
    experiment(2, 10, nn.ReLU(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
    experiment(2, 10, nn.Sigmoid(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
    experiment(2, 10, nn.Tanh(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
    experiment(2, 10, nn.LeakyReLU(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
    plotting(2, 10, model_name=model_name, exp_path=exp_path)


if __name__ == "__main__":
    model_name = "mleauto"
    for trial in range(num_trials):
        exp_path = get_dir(os.path.join(PATH_DICT[f"{model_name}_exp"], f"exp_{trial}"))
        experiments(model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
