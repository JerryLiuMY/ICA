from global_settings import PATH_DICT
from params.params import num_trials
import os


# if __name__ == "__main__":
#     from main import summarize
#     from main import experiment
#     from torch import nn
#     model_name = "mleauto"
#     exp_path = PATH_DICT[model_name]
#     experiment(2, 10, nn.ReLU(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
#     experiment(2, 10, nn.Sigmoid(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
#     experiment(2, 10, nn.Tanh(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
#     experiment(2, 10, nn.LeakyReLU(), model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
#     summarize(2, 10, model_name=model_name, exp_path=exp_path)


if __name__ == "__main__":
    from main import experiments
    model_name = "vae"
    for trial in range(num_trials):
        exp_path = os.path.join(PATH_DICT[f"{model_name}_exp"], f"trial_{trial}")
        if not os.path.isdir(exp_path):
            os.mkdir(exp_path)
            experiments(model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
