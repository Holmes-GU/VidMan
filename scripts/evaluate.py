import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra

from omegaconf import OmegaConf

from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from utils.calvin_utils import print_and_save

from utils.preprocess import PreProcess

import argparse
from opensora.utils.config_utils import (

    parse_configs,

)

from opensora.registry import MODELS, build_module
logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env

def evaluate_policy(model, env, eval_sr_path, eval_result_path, ep_len, num_sequences, num_procs, procs_id, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]
    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)

        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    unfinished = 0
    for step in range(ep_len):
        if unfinished == 0:
            action = model.step(obs, lang_annotation)
            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='training config')
    args = parser.parse_args()
    return args

def main():
    cfg = parse_configs(training=False)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    preprocessor = PreProcess(
        **cfg.data.preprocess,
        device=device,
    )
    policy = build_module(cfg.model, MODELS, device=device,
                          sequence_length=cfg['common']['sequence_length'],
                          act_dim=cfg['common']['act_dim'],
                          chunk_size=cfg['common']['chunk_size'])
    #TODO load model
    save_model = os.path.join(cfg['training']['save_path'], 'policy_{}.pth'.format(cfg['training']['load_epoch']))
    if os.path.isfile(save_model):
        missing_keys, unexpected_keys = policy.net.load_state_dict(torch.load(save_model), strict=False)
        acc.print('load', save_model, 'missing keys', missing_keys, "unexpected_keys", unexpected_keys)
        ema = os.path.join(cfg['training']['save_path'], 'ema_{}.pth'.format(cfg['training']['load_epoch']))
        missing_keys, unexpected_keys = policy.ema_net.load_state_dict(torch.load(ema), strict=False)
        acc.print('load', ema, 'missing keys', missing_keys, "unexpected_keys", unexpected_keys)
    policy.net, policy.ema_net = acc.prepare(
        policy.net,
        policy.ema_net,
        device_placement=[True, True],
    )
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}
    eval_dir = os.path.join(cfg['training']['save_path'], f'eval{torch.cuda.current_device()}/')
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env('./fake_dataset', observation_space, device)
    eva = GR1CalvinEvaluation(policy, cfg, preprocessor, device, data_interval=cfg['data']['dataset']['data_interval'], exec_action_step=cfg['common']['exec_action_step'])
    policy.ema_net.eval()
    avg_reward = torch.tensor(evaluate_policy(
        eva, 
        env,
        os.path.join(cfg['training']['save_path'],'success_rate.txt'),
        os.path.join(cfg['training']['save_path'],'result.txt'),
        cfg['common']['ep_len'],
        cfg['common']['num_sequences'],
        acc.num_processes,
        acc.process_index,
        eval_dir,
        debug=cfg['common']['record_evaluation_video'],
    )).float().mean().to(device)
    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)

if __name__ == "__main__":
    main()