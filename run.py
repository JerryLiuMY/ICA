from global_settings import PATH_DICT
from global_settings import get_dir
from main import run_experiment_multi
from main import run_experiments
from params.params import num_trials
from params.params import get_li_m_n
from params.params import get_iter_m_n
import os


# if __name__ == "__main__":
#     m, n = 1, 2
#     model_name = "vae"
#     exp_path = get_dir(PATH_DICT[model_name])
#
#     run_experiment_multi(m, n, model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)


if __name__ == "__main__":
    model_name = "vae"
    m_n_li, m_li_n = get_li_m_n(model_name)
    iter_m_n = get_iter_m_n(m_n_li, m_li_n)

    for trial in range(num_trials):
        exp_path = get_dir(os.path.join(PATH_DICT[f"{model_name}_exp"], f"trial_{trial}"))
        run_experiments(iter_m_n, model_name=model_name, exp_path=exp_path, train_s2=False, decoder_dgp=True)
