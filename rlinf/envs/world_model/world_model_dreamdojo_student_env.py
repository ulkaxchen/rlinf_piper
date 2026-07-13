# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Incremental DreamDojo student world-model environment for Piper."""

from __future__ import annotations

import gc
import os

import numpy as np
import torch

from rlinf.envs.world_model.world_model_dreamdojo_env import (
    DreamDojoEnv,
    _as_bool,
)

__all__ = ["DreamDojoStudentEnv"]


class DreamDojoStudentEnv(DreamDojoEnv):
    """Advance the causal DreamDojo student by one latent per env step.

    A reset first warms the native streaming state with one initial image and
    twelve demonstration-conditioned DreamDojo actions, producing twelve RGB
    frames. Subsequent ``chunk_step`` calls consume twelve 30 Hz policy actions,
    stride-sample them to four 10 Hz DreamDojo actions, and generate one latent
    (four RGB frames). The last nine RGB frames and last eight DreamDojo actions
    reconstruct the three-latent causal cache on every later call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.student_unload_on_offload = _as_bool(
            self.cfg.get("student_unload_on_offload", True)
        )
        self.student_decode_dit_offload = _as_bool(
            self.cfg.get("student_decode_dit_offload", True)
        )
        self.student_actions_per_latent = int(
            self.cfg.get("student_actions_per_latent", 4)
        )
        self.student_cache_latents = int(self.cfg.get("student_cache_latents", 3))
        self.student_condition_fps = float(self.cfg.get("student_condition_fps", 4.0))
        self.student_context_pixel_frames = (
            self.student_cache_latents - 1
        ) * self.student_actions_per_latent + 1
        self.student_history_actions = (
            self.student_cache_latents - 1
        ) * self.student_actions_per_latent
        self.student_bootstrap_enabled = _as_bool(
            self.cfg.get("student_bootstrap_enabled", True)
        )
        self.student_bootstrap_action_key = str(
            self.cfg.get("student_bootstrap_action_key", "abs_action")
        )
        self.student_bootstrap_policy_actions = int(
            self.cfg.get(
                "student_bootstrap_policy_actions",
                self.student_cache_latents
                * self.student_actions_per_latent
                * self.action_stride,
            )
        )
        self.student_bootstrap_model_actions = (
            self.student_bootstrap_policy_actions // self.action_stride
        )
        self.student_bootstrap_complete = False

        self._validate_student_temporal_contract()
        self._student_condition_frames: torch.Tensor | None = None
        self._student_action_history: torch.Tensor | None = None

    def _build_pipeline(self):
        """Load the self-forcing model from a local DCP checkpoint."""
        self._ensure_dreamdojo_on_path()
        self._validate_distilled_student_inputs()

        if _as_bool(self.cfg.get("dreamdojo_torch_compile", False)):
            raise ValueError(
                "DreamDojoStudentEnv currently requires "
                "dreamdojo_torch_compile=False because each chunk rebuilds its "
                "three-latent KV context for single-GPU offload compatibility."
            )

        cr1_embeddings_path = self.cr1_embeddings_path
        if cr1_embeddings_path in (None, "", "null"):
            raise ValueError(
                "DreamDojoStudentEnv requires cr1_embeddings_path or "
                "text_embedding_cache_path."
            )
        if not os.path.isfile(os.path.expanduser(str(cr1_embeddings_path))):
            raise FileNotFoundError(
                f"DreamDojo student CR1 embedding cache not found: "
                f"{cr1_embeddings_path}"
            )

        from cosmos_predict2._src.predict2.interactive.inference import (
            action_video2world,
        )

        # The fake-score network is a distillation-only training component. It
        # is not present in the streaming inference path, so avoid allocating
        # another 2B network during every serial onload.
        experiment_opts = ["model.config.net_fake_score=null"]
        experiment_opts.extend(self.cfg.get("student_experiment_opts", []))
        original_loader = action_video2world.load_model_from_checkpoint

        def _load_student_checkpoint(*args, **kwargs):
            kwargs["experiment_opts"] = (
                list(kwargs.get("experiment_opts", [])) + experiment_opts
            )
            return original_loader(*args, **kwargs)

        # ActionStreamingInference does not expose experiment_opts. Patch only
        # its module-local loader during construction, then immediately restore
        # it so teacher and other DreamDojo environments remain unaffected.
        action_video2world.load_model_from_checkpoint = _load_student_checkpoint
        try:
            pipe = action_video2world.ActionStreamingInference(
                config_path=self.cfg.config_file,
                experiment_name=self.cfg.experiment,
                ckpt_path=self.cfg.dreamdojo_ckpt_path,
                s3_credential_path=self.cfg.get(
                    "s3_credential_path", "credentials/s3_checkpoint.secret"
                ),
                cr1_embeddings_path=cr1_embeddings_path,
                context_parallel_size=int(
                    self.cfg.get("dreamdojo_context_parallel_size", 1)
                ),
                enable_fsdp=_as_bool(self.cfg.get("dreamdojo_enable_fsdp", False)),
                torch_compile=False,
            )
        finally:
            action_video2world.load_model_from_checkpoint = original_loader

        # The self-forcing training config instantiates teacher and fake-score
        # networks alongside the student. Streaming inference only calls
        # ``model.net``; retaining the other two 2B networks would waste many
        # gigabytes when this environment is offloaded to host memory.
        for attr in ("net_teacher", "net_fake_score", "net_discriminator_head"):
            if hasattr(pipe.model, attr):
                setattr(pipe.model, attr, None)
        gc.collect()
        self._clear_accelerator_cache()
        return pipe

    def _validate_student_temporal_contract(self):
        if self.student_actions_per_latent != 4:
            raise ValueError(
                "The current DreamDojo student checkpoint requires "
                "student_actions_per_latent=4."
            )
        if self.student_cache_latents != 3:
            raise ValueError(
                "The current DreamDojo student checkpoint requires "
                "student_cache_latents=3."
            )
        if self.gen_frames != self.student_actions_per_latent:
            raise ValueError(
                "DreamDojoStudentEnv must generate exactly one latent per "
                "chunk_step: chunk // action_stride must equal "
                f"{self.student_actions_per_latent}, got {self.chunk} // "
                f"{self.action_stride} = {self.gen_frames}."
            )
        expected_bootstrap_actions = (
            self.student_cache_latents * self.student_actions_per_latent
        )
        if (
            self.student_bootstrap_policy_actions <= 0
            or self.student_bootstrap_policy_actions % self.action_stride != 0
            or self.student_bootstrap_model_actions != expected_bootstrap_actions
        ):
            raise ValueError(
                "DreamDojoStudentEnv bootstrap must provide exactly "
                f"{expected_bootstrap_actions} 10Hz model actions, i.e. "
                f"{expected_bootstrap_actions * self.action_stride} 30Hz policy "
                "actions."
            )

        model = self.pipe.model
        model_actions_per_latent = int(model.net._num_action_per_latent_frame)
        if model_actions_per_latent != self.student_actions_per_latent:
            raise ValueError(
                "DreamDojo network action/latent ratio does not match the env: "
                f"model={model_actions_per_latent}, "
                f"env={self.student_actions_per_latent}."
            )
        model_cache_latents = int(model.config.cache_frame_size)
        if model_cache_latents != self.student_cache_latents:
            raise ValueError(
                "DreamDojo network cache size does not match the env: "
                f"model={model_cache_latents}, env={self.student_cache_latents}."
            )

    @torch.no_grad()
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.student_bootstrap_complete = False
        if self.student_bootstrap_enabled:
            self._bootstrap_student_context()
            self.student_bootstrap_complete = True
            return self._wrap_obs(), info

        # Legacy fallback for callers without a demonstration action prefix.
        # It is intentionally opt-in because repeated copies of the initial
        # image are not equivalent to the native 1-frame -> 12-frame warmup.
        initial = self.current_obs.detach().cpu().permute(0, 3, 1, 2)
        self._student_condition_frames = initial.unsqueeze(2).repeat(
            1, 1, self.student_context_pixel_frames, 1, 1
        )
        self._student_action_history = torch.zeros(
            self.num_envs,
            self.student_history_actions,
            self.model_action_dim,
            dtype=torch.float32,
        )
        return obs, info

    def _load_student_bootstrap_policy_actions(self) -> torch.Tensor:
        """Load the 36-action 30Hz demonstration prefix for each reset env."""
        episode_indices = self._last_reset_episode_indices
        if episode_indices is None or len(episode_indices) != self.num_envs:
            raise RuntimeError(
                "DreamDojoStudentEnv did not receive reset episode indices for "
                "streaming warmup."
            )

        actions_per_env = []
        for env_idx, episode_idx in enumerate(episode_indices):
            trajectory = self.dataset._load_trajectory(int(episode_idx))
            if len(trajectory) < self.student_bootstrap_policy_actions:
                raise ValueError(
                    "DreamDojo student native warmup needs at least "
                    f"{self.student_bootstrap_policy_actions} frames/actions in "
                    f"{self.dataset.npy_files[int(episode_idx)]}, but found "
                    f"{len(trajectory)}. Re-export initial frames with "
                    "convert_piper_to_initial_npy.py --frames-per-file 36."
                )

            prefix = []
            for action_idx in range(self.student_bootstrap_policy_actions):
                frame = trajectory[action_idx]
                if self.student_bootstrap_action_key not in frame:
                    raise KeyError(
                        "DreamDojo student warmup requires frame key "
                        f"{self.student_bootstrap_action_key!r} in "
                        f"{self.dataset.npy_files[int(episode_idx)]}."
                    )
                action = np.asarray(
                    frame[self.student_bootstrap_action_key], dtype=np.float32
                ).reshape(-1)
                if action.size < self.piper_action_dim:
                    raise ValueError(
                        "DreamDojo student warmup action dimension is too small: "
                        f"env={env_idx}, frame={action_idx}, got {action.size}, "
                        f"need {self.piper_action_dim}."
                    )
                prefix.append(torch.from_numpy(action[: self.piper_action_dim]))
            actions_per_env.append(torch.stack(prefix, dim=0))
        return torch.stack(actions_per_env, dim=0)

    @torch.no_grad()
    def _generate_student_bootstrap_video(
        self,
        initial_frame: torch.Tensor,
        model_actions: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Generate the native one-frame-context, twelve-frame student prefix."""
        video = self.pipe.generate_action_streaming(
            video_path=initial_frame.permute(1, 2, 0).unsqueeze(0).cpu().numpy(),
            actions_np=model_actions.cpu().numpy(),
            resolution_hw=(self.gen_height, self.gen_width),
            num_steps=self.num_inference_steps,
            seed=seed,
            start_frame_idx=0,
            max_frames=self.student_bootstrap_model_actions + 1,
        )
        if video.ndim != 5 or video.shape[:2] != (1, 3):
            raise RuntimeError(
                "DreamDojo student bootstrap returned invalid video shape "
                f"{tuple(video.shape)}."
            )
        expected_frames = self.student_bootstrap_model_actions + 1
        if video.shape[2] < expected_frames:
            raise RuntimeError(
                "DreamDojo student bootstrap generated too few frames: "
                f"got {video.shape[2]}, need {expected_frames}."
            )
        return video[:, :, -expected_frames:].detach().cpu()

    @torch.no_grad()
    def _bootstrap_student_context(self) -> None:
        """Initialize the native 9-frame / 8-action streaming context.

        The warmup consumes dataset demonstration actions only. It establishes
        the world-model state before RL starts, so those actions and frames are
        deliberately absent from the policy trajectory and GRPO loss.
        """
        policy_actions = self._load_student_bootstrap_policy_actions()
        model_actions = policy_actions[:, :: self.action_stride]
        if model_actions.shape[1] != self.student_bootstrap_model_actions:
            raise RuntimeError(
                "Student bootstrap action stride produced "
                f"{model_actions.shape[1]} model actions; expected "
                f"{self.student_bootstrap_model_actions}."
            )
        if self.policy_action_format == "absolute":
            model_actions = self._absolute_to_dreamdojo_delta(model_actions)

        condition_frames = []
        action_history = []
        final_frames = []
        for env_idx in range(self.num_envs):
            cosmos_actions = self._build_model_action(model_actions[env_idx])
            initial = self.current_obs[env_idx].detach().cpu().permute(2, 0, 1)
            video = self._generate_student_bootstrap_video(
                initial,
                cosmos_actions,
                self.seed_base + env_idx,
            )
            pixels = ((video[0] + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
            generated = pixels[:, 1:]
            condition_frames.append(
                generated[:, -self.student_context_pixel_frames :].contiguous()
            )
            action_history.append(
                cosmos_actions[-self.student_history_actions :].contiguous()
            )
            final_frames.append(generated[:, -1].permute(1, 2, 0).contiguous())

        self._student_condition_frames = torch.stack(condition_frames, dim=0)
        self._student_action_history = torch.stack(action_history, dim=0)
        self.current_obs = torch.stack(final_frames, dim=0).to(self.device)
        self.current_states = policy_actions[:, -1].to(self.device)
        self._last_action_state = policy_actions[:, -1].clone()
        # Bootstrap establishes context before RL; do not expose its reward.
        self.last_chunk_frames = None

    def _build_student_condition(self, env_idx: int, current_actions: torch.Tensor):
        """Encode the rolling pixel context and construct action conditions."""
        from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import (
            ActionConditionedCondition,
        )
        from cosmos_predict2._src.predict2.conditioner import DataType

        cond_frames = self._student_condition_frames[env_idx : env_idx + 1]
        zero_tail = torch.zeros_like(cond_frames[:, :, :1]).repeat(
            1, 1, self.student_actions_per_latent, 1, 1
        )
        video_input = torch.cat([cond_frames, zero_tail], dim=2)

        history = self._student_action_history[env_idx]
        all_actions = torch.cat([history, current_actions], dim=0).unsqueeze(0)
        data_batch = self.pipe._prepare_data_batch(
            video_b_c_t_h_w=video_input,
            actions_np=all_actions.numpy(),
            fps=self.student_condition_fps,
            num_latent_conditional_frames=0,
        )

        model = self.pipe.model
        model._normalize_video_databatch_inplace(data_batch)
        model._augment_image_dim_inplace(data_batch)

        text = self.pipe.t5_text_embeddings_cpu.to(
            device=model.tensor_kwargs["device"], dtype=torch.bfloat16
        )
        data_batch["t5_text_embeddings"] = text
        data_batch["t5_text_mask"] = torch.ones(
            (text.shape[0], text.shape[1]),
            device=text.device,
            dtype=torch.bfloat16,
        )
        for key, value in list(data_batch.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                data_batch[key] = value.to(dtype=model.tensor_kwargs["dtype"])

        _, x0, condition, _ = model.get_data_and_condition(data_batch)
        x0 = x0.to(dtype=model.tensor_kwargs["dtype"])
        condition = condition.edit_data_type(DataType.VIDEO)
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=None,
            random_max_num_conditional_frames=None,
            num_conditional_frames=0,
        )

        condition_dict = condition.to_dict()
        condition_dict["action"] = current_actions.unsqueeze(0).to(
            device=model.tensor_kwargs["device"],
            dtype=model.tensor_kwargs["dtype"],
        )
        generation_condition = ActionConditionedCondition(**condition_dict)
        return x0, condition, generation_condition

    @torch.no_grad()
    def _generate_one_latent(
        self, env_idx: int, current_actions: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """Generate and decode one latent, returning four RGB frames."""
        from cosmos_predict2._src.imaginaire.utils import misc
        from cosmos_predict2._src.predict2.interactive.networks.utils import (
            make_network_kv_cache,
        )
        from cosmos_predict2._src.predict2.utils.kv_cache import VideoSeqPos

        model = self.pipe.model
        x0, prefill_condition, generation_condition = self._build_student_condition(
            env_idx, current_actions
        )
        if x0.shape[2] != self.student_cache_latents + 1:
            raise RuntimeError(
                "Expected the 13-frame student input to encode to four latents, "
                f"got latent shape {tuple(x0.shape)}."
            )

        context_latents = x0[:, :, : self.student_cache_latents]
        _, channels, _, latent_h, latent_w = x0.shape
        noise = misc.arch_invariant_rand(
            (1, channels, 1, latent_h, latent_w),
            torch.float32,
            model.tensor_kwargs["device"],
            seed,
        ).to(**model.tensor_kwargs)

        token_h = latent_h // model.net.patch_spatial
        token_w = latent_w // model.net.patch_spatial
        full_video_pos = VideoSeqPos(
            T=self.student_cache_latents + 1,
            H=token_h,
            W=token_w,
        )
        make_network_kv_cache(
            model.net,
            max_cache_size=self.student_cache_latents,
            stateless=False,
        )
        for frame_idx in range(self.student_cache_latents):
            model.update_kv_cache(
                context_latents[:, :, frame_idx : frame_idx + 1],
                prefill_condition,
                full_video_pos,
                frame_idx,
                self.student_cache_latents,
                1,
                model.tensor_kwargs["device"],
            )

        num_steps = min(
            max(1, int(self.num_inference_steps)),
            len(model.config.selected_sampling_time),
        )
        predicted_latent = model.generate_next_frame(
            generation_condition,
            noise,
            self.student_cache_latents,
            self.student_cache_latents,
            full_video_pos=full_video_pos,
            n_steps=num_steps,
        )
        decode_latents = torch.cat([context_latents, predicted_latent], dim=2)

        dit_offloaded = False
        if self.student_decode_dit_offload and self.device.type == "cuda":
            # Student generation is over, so none of the DiT KV entries are
            # needed by the causal VAE decoder. Reset them before moving the
            # 2B student to CPU; otherwise DiT weights and the full-resolution
            # VAE decode peak overlap on a 32GB card.
            make_network_kv_cache(
                model.net,
                max_cache_size=self.student_cache_latents,
                stateless=False,
            )
            del x0, prefill_condition, generation_condition, noise
            model.net = model.net.to("cpu")
            dit_offloaded = True
            gc.collect()
            self._clear_accelerator_cache()

        try:
            decoded = self.pipe._decode(decode_latents).clip(min=-1, max=1)
            if decoded.shape[2] < self.student_actions_per_latent:
                raise RuntimeError(
                    "DreamDojo student decoded fewer than four frames: "
                    f"{tuple(decoded.shape)}"
                )
            result = decoded[:, :, -self.student_actions_per_latent :].cpu()
            del decoded
            return result
        finally:
            if dit_offloaded:
                model.net = model.net.to(self.device)
                self._clear_accelerator_cache()

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        """Consume 12 policy actions and advance the student by one latent."""
        if actions.shape[:2] != (self.num_envs, self.chunk):
            raise ValueError(
                "DreamDojoStudentEnv expected action shape "
                f"[{self.num_envs}, {self.chunk}, ...], got {tuple(actions.shape)}."
            )
        actions = torch.as_tensor(actions).detach().cpu()[:, :: self.action_stride]
        if actions.shape[1] != self.student_actions_per_latent:
            raise RuntimeError(
                f"Stride-sampled action length must be 4, got {actions.shape[1]}."
            )
        self._last_action_state = actions[:, -1, : self.piper_action_dim].clone()

        if self.policy_action_format == "absolute":
            actions = self._absolute_to_dreamdojo_delta(actions)

        new_last_frames = []
        new_chunks = []
        for env_idx in range(self.num_envs):
            model_actions = self._build_model_action(actions[env_idx])
            generated = self._generate_one_latent(
                env_idx,
                model_actions,
                self.seed_base + self.elapsed_steps + env_idx,
            )
            pixels = ((generated + 1.0) / 2.0).clamp(0, 1)
            pixels = (pixels[0] * 255.0).to(torch.uint8)
            new_chunks.append(pixels.permute(1, 0, 2, 3).contiguous())
            new_last_frames.append(pixels[:, -1].permute(1, 2, 0).contiguous())

            context = torch.cat(
                [self._student_condition_frames[env_idx], pixels], dim=1
            )
            self._student_condition_frames[env_idx] = context[
                :, -self.student_context_pixel_frames :
            ]
            action_history = torch.cat(
                [self._student_action_history[env_idx], model_actions], dim=0
            )
            self._student_action_history[env_idx] = action_history[
                -self.student_history_actions :
            ]

            del generated
            self._clear_accelerator_cache()

        self.current_obs = torch.stack(new_last_frames, dim=0).to(self.device)
        self.last_chunk_frames = torch.stack(new_chunks, dim=0).to(self.device)

    def offload(self):
        """Release the student pipeline completely between serial GPU phases."""
        if self._is_offloaded:
            return
        if self.student_unload_on_offload and self.pipe is not None:
            pipe = self.pipe
            self.pipe = None
            try:
                pipe.cleanup()
            finally:
                del pipe
                gc.collect()
                self._clear_accelerator_cache()
        super().offload()

    def onload(self):
        """Reload the student checkpoint when the environment becomes active."""
        if not self._is_offloaded:
            return
        if self.pipe is None:
            self.pipe = self._build_pipeline()
            self._validate_student_temporal_contract()
        super().onload()
