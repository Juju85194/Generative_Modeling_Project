import yaml
from src.h_fcn import *
from src.utils import init_model
from src.eval import evaluate_model
from src.utils import display_logs, save_results
import torch
import warnings
warnings.filterwarnings("ignore")

map_fcn = {"Inpainting": Inpainting,
           "Blurring": Blurring,
           "Linear": Linear,
           "Superres": Superres}

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # LOAD PARAMETERS  
    run_dps = config["run_dps"]

    path = config["dataset_path"]

    figs_path = config["figs_path"]

    results_path = config["results_path"]

    eta = config["eta"]

    num_img = config["num_images"]

    num_ddim_steps = config['num_ddim_steps']

    num_samples = config["num_samples"]

    num_diffusion_timesteps = config["num_diffusion_timesteps"]

    model = init_model()

    for fcn_name in config["experiences"]:
        h_fcn = map_fcn[fcn_name]

        out, logs = evaluate_model(model, h_fcn, path, num_images=num_img,
                                   num_ddim_steps=num_ddim_steps, num_samples=num_samples,
                                   eta=eta, awd=True, show_steps=False,
                                   num_diffusion_timesteps=num_diffusion_timesteps,
                                   device=device, run_dps=run_dps, keep_logs=True)

        path_pgdm = "/pgdm_" + fcn_name + "_" + num_ddim_steps + "_" + num_img

        if run_dps:
            display_logs(logs[0], figs_path + path_pgdm + ".png", False)
            path_dps = "/dps_" + fcn_name + "_" + num_ddim_steps + "_" + num_img
            display_logs(logs[1], figs_path + path_dps + ".png", False)
        else:
            display_logs(logs, figs_path + path_pgdm + ".png", False)

        save_results(out, path=results_path, fcn=fcn_name,
                     n_ddim=num_ddim_steps, n_img=num_img)
