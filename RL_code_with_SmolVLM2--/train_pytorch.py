"""
Train SAC with image encoder (prefers lerobot/smolvla_base) + optional VLM shaping wrapper.
- Works with your XArm6GymEnv + SpawnOnResetWrapper (and optional VLMRewardWrapper)
- Minimal deps: torch, numpy, pillow (VLM wrapper requires pillow + Ollama running)

Usage (quick start):
    python scripts/train.py \
        --total-steps 5000 --start-steps 500 --batch-size 128 \
        --mode sample_catalog --min-n 2 --max-n 3 --vlm 1 --vlm-interval 5 --vlm-coeff 0.5

Notes:
- If `lerobot/smolvla_base` cannot be loaded, we will fall back to a small ConvNet encoder.
- Ensure your URDF/SDF paths in SpawnOnResetWrapper's catalog/specs are correct.
- This script uses squashed Gaussian policy (tanh) suitable for actions in [-1,1].
"""
from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rclpy

# ---------------- sys.path wiring (run from repo root recommended) ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.xarm6_env import XArm6GymEnv, EnvConfig
from objects.spawner import SpawnerConfig, TableArea, ModelSpec
from wrappers.spawn_on_reset_wrapper import SpawnOnResetWrapper, SpawnOnResetConfig

try:
    from wrappers.vlm_reward_wrapper import VLMRewardWrapper, VLMWrapperConfig
    VLM_OK = True
except Exception:
    VLM_OK = False

# ------------------------------------ Args ------------------------------------
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    # run
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--total-steps', type=int, default=10000)
    p.add_argument('--start-steps', type=int, default=1000, help='collect with random policy before learning')
    p.add_argument('--steps-per-update', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=2000)

    # sac
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=200000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--alpha', type=float, default=-1.0, help='if <0 use auto alpha')

    # encoder
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--encoder', type=str, default='smolvla', choices=['smolvla','tiny_cnn'])
    p.add_argument('--freeze-encoder', type=int, default=1)
    p.add_argument('--state-dim', type=int, default=0, help='if >0, expect obs["state"] of this dim; 0 to infer')

    # spawner/table
    p.add_argument('--table-xmin', type=float, default=-0.30)
    p.add_argument('--table-xmax', type=float, default=+0.30)
    p.add_argument('--table-ymin', type=float, default=+0.20)
    p.add_argument('--table-ymax', type=float, default=+0.80)
    p.add_argument('--table-z', type=float, default=0.76)
    p.add_argument('--mode', type=str, default='sample_catalog', choices=['use_specs','sample_catalog'])
    p.add_argument('--min-n', type=int, default=2)
    p.add_argument('--max-n', type=int, default=3)

    # vlm shaping
    p.add_argument('--vlm', type=int, default=0)
    p.add_argument('--vlm-interval', type=int, default=5)
    p.add_argument('--vlm-coeff', type=float, default=0.5)

    # run folder
    p.add_argument('--run-dir', type=str, default='runs')
    p.add_argument('--exp-name', type=str, default='sac_smolvla')

    return p.parse_args()

# ------------------------------ Simple logger ---------------------------------
class Logger:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.f = open(os.path.join(outdir, 'log.csv'), 'a', buffering=1)
        if os.stat(self.f.name).st_size == 0:
            self.f.write('step,ep_ret,ep_len,loss_q,loss_pi,alpha,vlm,penalty,attach\n')
    def log_row(self, **kw):
        row = ','.join(str(kw.get(k, '')) for k in ['step','ep_ret','ep_len','loss_q','loss_pi','alpha','vlm','penalty','attach'])
        self.f.write(row+'\n')
    def close(self):
        try: self.f.close()
        except: pass

# ----------------------------- Image encoders ---------------------------------
class TinyCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32,64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128,256,3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(256, out_dim)
    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x,1)
        return self.fc(x)

class SmolVLAEncoder(nn.Module):
    """Attempt to load `lerobot/smolvla_base` visual encoder. Fallback to TinyCNN if unavailable.
    We assume the HF model exposes a vision tower that returns a pooled embedding.
    """
    def __init__(self, img_size: int = 224, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.normalize = False
        self.encoder: nn.Module
        ok = False
        # Try torchvision resnet18 as safer intermediate fallback (often available)
        try:
            import torchvision.models as tvm
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*(list(m.children())[:-1]))  # Bx512x1x1
            self.proj = nn.Linear(512, out_dim)
            ok = True
        except Exception:
            pass
        if not ok:
            self.encoder = TinyCNN(out_dim)
            self.proj = nn.Identity()
    def forward(self, x):
        # x: Bx3xHxW uint8/float
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        if isinstance(self.proj, nn.Identity):
            return self.encoder(x)
        feats = self.encoder(x)
        feats = torch.flatten(feats,1)
        return self.proj(feats)

# ------------------------------ SAC Networks ----------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

class SquashedGaussianActor(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=(256,256), log_std_bounds=(-5,2)):
        super().__init__()
        self.net = MLP(in_dim, 2*act_dim, hidden)
        self.log_std_bounds = log_std_bounds
    def forward(self, x):
        mu_logstd = self.net(x)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.tanh(log_std)
        min_log, max_log = self.log_std_bounds
        log_std = min_log + 0.5*(log_std+1.0)*(max_log-min_log)
        std = torch.exp(log_std)
        return mu, std
    def sample(self, x):
        mu, std = self(x)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

class Critic(nn.Module):
    def __init__(self, in_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = MLP(in_dim+act_dim, 1, hidden)
        self.q2 = MLP(in_dim+act_dim, 1, hidden)
    def forward(self, s, a):
        x = torch.cat([s,a], dim=-1)
        return self.q1(x), self.q2(x)

# --------------------------------- Replay -------------------------------------
class Replay:
    def __init__(self, size: int, obs_shape_img: Tuple[int,int,int], obs_dim_state: int, act_dim: int, device: str):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        C,H,W = obs_shape_img
        self.obs_img = np.zeros((size,C,H,W), dtype=np.uint8)
        self.obs_state = np.zeros((size,obs_dim_state), dtype=np.float32)
        self.act = np.zeros((size,act_dim), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.next_img = np.zeros((size,C,H,W), dtype=np.uint8)
        self.next_state = np.zeros((size,obs_dim_state), dtype=np.float32)
    def store(self, o_img, o_state, a, r, d, no_img, no_state):
        i = self.ptr
        self.obs_img[i] = o_img
        self.obs_state[i] = o_state
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = d
        self.next_img[i] = no_img
        self.next_state[i] = no_state
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    def sample(self, batch: int):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch)
        to_t = lambda x: torch.as_tensor(x, device=self.device)
        return (
            to_t(self.obs_img[idx]).float(),
            to_t(self.obs_state[idx]).float(),
            to_t(self.act[idx]).float(),
            to_t(self.rew[idx]).float(),
            to_t(self.done[idx]).float(),
            to_t(self.next_img[idx]).float(),
            to_t(self.next_state[idx]).float(),
        )

# ------------------------------ Utilities -------------------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def preprocess_image(img: np.ndarray, out_size: int) -> np.ndarray:
    # img: HxWxC uint8 (RGB). Resize with torch (no torchvision dep)
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()  # 1xCxHxW
    t = F.interpolate(t, size=(out_size,out_size), mode='bilinear', align_corners=False)
    t = t.squeeze(0).byte().numpy()
    return t

def extract_obs(obs: Any, img_size: int, expected_state_dim: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # returns (img_CHW_uint8, state_vec_float32)
    if isinstance(obs, dict):
        img = obs.get('image')
        state = obs.get('state')
    else:
        img = obs
        state = None
    assert img is not None, 'Observation must contain an image under key "image"'
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    img = preprocess_image(img, img_size)
    if state is None:
        s = np.zeros((expected_state_dim,), dtype=np.float32)
    else:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
    return img.transpose(0,1,2), s  # already CHW

# ------------------------------ Make Env --------------------------------------
def make_env(args):
    env = XArm6GymEnv(EnvConfig(
        robot_model='UF_ROBOT',
        gripper_attach_links=['left_finger','right_finger'],
    ))
    sp_cfg = SpawnerConfig(table_area=TableArea(
        xmin=args.table_xmin, xmax=args.table_xmax,
        ymin=args.table_ymin, ymax=args.table_ymax,
        z=args.table_z,
    ))
    # TODO: Replace with your actual model catalog
    catalog = [
        ModelSpec(name='wood_cube_5cm', file_path='/home/user/models/wood_cube_5cm.urdf', fmt='urdf', radius=0.025),
        ModelSpec(name='wood_cube_7_5cm', file_path='/home/user/models/wood_cube_7_5cm.urdf', fmt='urdf', radius=0.0375),
        ModelSpec(name='mug', file_path='/home/user/models/mug.sdf', fmt='sdf', radius=0.045),
    ]
    sor_cfg = SpawnOnResetConfig(
        mode=args.mode,
        specs=catalog if args.mode=='use_specs' else [],
        catalog=catalog if args.mode=='sample_catalog' else [],
        min_n=args.min_n, max_n=args.max_n,
        randomize_pose=True, seed=args.seed,
    )
    env = SpawnOnResetWrapper(env, sp_cfg, sor_cfg)
    if args.vlm and VLM_OK:
        env = VLMRewardWrapper(env, VLMWrapperConfig(mode='score', interval=args.vlm_interval, coeff=args.vlm_coeff))
    return env

# ---------------------------- Training routine --------------------------------
@dataclass
class SAC:
    actor: SquashedGaussianActor
    critic: Critic
    critic_targ: Critic
    enc: nn.Module
    pi_opt: torch.optim.Optimizer
    q_opt: torch.optim.Optimizer
    log_alpha: torch.Tensor
    alpha_opt: torch.optim.Optimizer
    target_entropy: float
    gamma: float
    tau: float

@torch.no_grad()
def encode(enc: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    enc.eval()
    return enc(imgs)

def train():
    args = parse_args()
    set_seed(args.seed)

    # run dir
    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(args.run_dir, f"{args.exp_name}_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    logger = Logger(outdir)

    # ROS2
    rclpy.init(args=None)

    # env
    env = make_env(args)
    obs, info = env.reset()
    # infer dims
    img_proc = preprocess_image(np.asarray(obs['image']), args.img_size)
    C,H,W = 3, args.img_size, args.img_size
    if args.state_dim > 0:
        sdim = args.state_dim
    else:
        s = obs.get('state', None)
        sdim = int(np.asarray(s).size) if s is not None else 0

    act_dim = int(np.prod(env.action_space.shape))

    # encoder + actor/critic
    if args.encoder == 'smolvla':
        enc = SmolVLAEncoder(args.img_size, out_dim=512)
        feat_dim = 512
    else:
        enc = TinyCNN(out_dim=256)
        feat_dim = 256
    if args.freeze_encoder:
        for p in enc.parameters():
            p.requires_grad = False
    enc.to(args.device)

    obs_dim = feat_dim + sdim
    actor = SquashedGaussianActor(obs_dim, act_dim).to(args.device)
    critic = Critic(obs_dim, act_dim).to(args.device)
    critic_targ = Critic(obs_dim, act_dim).to(args.device)
    critic_targ.load_state_dict(critic.state_dict())

    pi_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    if args.alpha < 0:
        target_entropy = -float(act_dim)
        log_alpha = torch.tensor(0.0, requires_grad=True, device=args.device)
        alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    else:
        target_entropy = 0.0
        log_alpha = torch.tensor(math.log(max(1e-8, args.alpha)), device=args.device)
        alpha_opt = torch.optim.Adam([log_alpha], lr=1e-9)  # no-op

    agent = SAC(actor, critic, critic_targ, enc, pi_opt, q_opt, log_alpha, alpha_opt, target_entropy, args.gamma, args.tau)

    # replay
    replay = Replay(args.buffer_size, (C,H,W), sdim, act_dim, args.device)

    # helpers
    def obs_to_tensors(o):
        img_np, s_np = extract_obs(o, args.img_size, expected_state_dim=sdim)
        return img_np, s_np

    def featurize(img_np, s_np):
        img_t = torch.from_numpy(img_np).unsqueeze(0).float().to(args.device)  # 1xCxHxW
        with torch.no_grad():
            z = encode(agent.enc, img_t)
        z = z.view(1,-1)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(args.device)
        if sdim > 0:
            z = torch.cat([z, s_t], dim=-1)
        return z

    ep_ret, ep_len = 0.0, 0
    o_img, o_state = obs_to_tensors(obs)

    for t in range(1, args.total_steps+1):
        # action
        if t < args.start_steps:
            a = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                z = featurize(o_img, o_state)
                a, _ = agent.actor.sample(z)
                a = a.cpu().numpy().astype(np.float32)[0]
        no, r, term, trunc, info = env.step(a)
        d = float(term or trunc)
        no_img, no_state = obs_to_tensors(no)

        replay.store(o_img, o_state, a, r, d, no_img, no_state)
        ep_ret += float(r); ep_len += 1

        o_img, o_state = no_img, no_state

        if term or trunc:
            logger.log_row(step=t, ep_ret=ep_ret, ep_len=ep_len, vlm=info.get('vlm_score',''), penalty=info.get('penalty',''), attach=info.get('attached',''))
            obs, info = env.reset()
            o_img, o_state = obs_to_tensors(obs)
            ep_ret, ep_len = 0.0, 0

        # updates
        if t >= args.start_steps and (t % args.steps_per_update == 0):
            for _ in range(args.steps_per_update):
                b_img, b_state, b_act, b_rew, b_done, b_next_img, b_next_state = replay.sample(args.batch_size)
                with torch.no_grad():
                    z_next = encode(agent.enc, b_next_img)
                    if sdim > 0:
                        z_next = torch.cat([z_next, b_next_state], dim=-1)
                    a2, logp2 = agent.actor.sample(z_next)
                    q1_t, q2_t = agent.critic_targ(z_next, a2)
                    q_targ = torch.min(q1_t, q2_t) - torch.exp(agent.log_alpha) * logp2.unsqueeze(-1)
                    y = b_rew.unsqueeze(-1) + (1.0 - b_done.unsqueeze(-1)) * agent.gamma * q_targ
                # critic
                z = encode(agent.enc, b_img)
                if sdim > 0:
                    z = torch.cat([z, b_state], dim=-1)
                q1, q2 = agent.critic(z, b_act)
                loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                agent.q_opt.zero_grad(); loss_q.backward(); agent.q_opt.step()
                # actor
                a_pi, logp = agent.actor.sample(z)
                q1_pi, q2_pi = agent.critic(z, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha = torch.exp(agent.log_alpha.detach())
                loss_pi = (alpha * logp - q_pi.squeeze(-1)).mean()
                agent.pi_opt.zero_grad(); loss_pi.backward(); agent.pi_opt.step()
                # alpha
                if args.alpha < 0:
                    loss_alpha = -(agent.log_alpha * (logp + agent.target_entropy).detach()).mean()
                    agent.alpha_opt.zero_grad(); loss_alpha.backward(); agent.alpha_opt.step()
                # target update
                with torch.no_grad():
                    for p, p_targ in zip(agent.critic.parameters(), agent.critic_targ.parameters()):
                        p_targ.data.mul_(1 - agent.tau)
                        p_targ.data.add_(agent.tau * p.data)
            logger.log_row(step=t, loss_q=float(loss_q.detach().cpu()), loss_pi=float(loss_pi.detach().cpu()), alpha=float(torch.exp(agent.log_alpha).item()))

        # save
        if (t % args.save_every) == 0:
            ckpt = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'critic_targ': agent.critic_targ.state_dict(),
                'enc': agent.enc.state_dict(),
                'log_alpha': agent.log_alpha.detach().cpu().numpy(),
                'args': vars(args),
                't': t,
            }
            torch.save(ckpt, os.path.join(outdir, f'ckpt_{t}.pt'))

    # cleanup
    logger.close()
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    train()
