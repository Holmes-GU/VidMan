import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import clip



from calvin_agent.models.calvin_base_model import CalvinBaseModel

class GR1CalvinEvaluation(CalvinBaseModel):
    def __init__(self,
                 policy,
                 cfg,
                 preprocessor,
                 device,
                 data_interval=1,
                 exec_action_step=-1
    ):
        self.tokenizer = clip.tokenize
        self.seq_len = cfg['common']['sequence_length']
        self.chunk_size = cfg['common']['chunk_size']
        self.test_chunk_size = cfg['common']['test_chunk_size']
        self.act_dim = cfg['common']['act_dim']
        self.state_dim = cfg['model']['net']['state_dim']
        self.device = device
        self.use_hand_rgb=True

        # Preprocess
        self.preprocessor = preprocessor 
        self.policy = policy
        self.data_interval = data_interval
        self.cache_interval = 0
        self.exec_action_step = exec_action_step
    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0
        self.cache_interval = 0
        
    def step(self, obs, goal):
        """Step function."""
        # Language
        text = goal
        tokenized_text = self.tokenizer(text)

        # RGB
        rgb = rearrange(torch.from_numpy(obs['rgb_obs']['rgb_static']), 'h w c -> c h w')
        hand_rgb = rearrange(torch.from_numpy(obs['rgb_obs']['rgb_gripper']), 'h w c -> c h w')

        # State
        state = obs['robot_obs']
        arm_state = state[:6]
        gripper_state = state[-1]
        state = torch.from_numpy(np.hstack([arm_state, gripper_state]))

        if self.cache_interval % self.data_interval == 0:
            self.rgb_list.append(rgb)
            self.hand_rgb_list.append(hand_rgb)
            self.state_list.append(state)
            self.cache_interval = 0
        self.cache_interval += 1

        buffer_len = len(self.rgb_list)
        if buffer_len > self.seq_len:
            self.rgb_list.pop(0)
            self.hand_rgb_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_list) == self.seq_len
            assert len(self.hand_rgb_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_list)

        # Static RGB
        c, h, w = rgb.shape
        rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_tensor = torch.stack(self.rgb_list, dim=0)  # (t, c, h, w)
        rgb_data[0, :buffer_len] = rgb_tensor

        # Hand RGB
        c, h, w = hand_rgb.shape
        hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0)  # (t, c, h, w)
        hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # State
        state_tensor = torch.stack(self.state_list, dim=0)  # (l, act_dim)
        gripper_state_data = - torch.ones((1, self.seq_len)).float()
        gripper_state_data[0, :buffer_len] = state_tensor[:, 6]
        gripper_state_data = (gripper_state_data + 1.0) / 2
        gripper_state_data = gripper_state_data.long()
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).float()  # (1, t, 2)
        arm_state_data = torch.zeros((1, self.seq_len, self.act_dim - 1)).float()  # (1, t, act_dim - 1)
        arm_state_data[0, :buffer_len] = state_tensor[:, :6]

        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1

        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        rgb_data = rgb_data.to(self.device)
        hand_rgb_data = hand_rgb_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)

        rgb_data, hand_rgb_data = self.preprocessor.rgb_process(rgb_data, hand_rgb_data, train=False)
        with torch.no_grad():
            prediction = self.policy.infer(
                rgb=rgb_data, 
                hand_rgb=hand_rgb_data,
                state=state_data,
                language=tokenized_text,
                attention_mask=attention_mask
        )

        # Arm action
        arm_action_preds = prediction[..., :-1]  # (1, t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds.view(-1, self.chunk_size, self.act_dim - 1)  # (t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]

        # Gripper action
        gripper_action_preds = prediction[..., -1:]  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, self.chunk_size, 1)  # (t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]

        # Use the last action
        if self.exec_action_step == -1:
            arm_action_pred = arm_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
            gripper_action_pred = gripper_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, 1)

        gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        gripper_action_pred = gripper_action_pred > 0.5
        gripper_action_pred = gripper_action_pred.int().float()
        gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (act_dim,)
        action_pred = action_pred.detach().cpu()

        self.rollout_step_counter += 1
    
        return action_pred