# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Incremental DreamDojo distilled-student environment for Piper."""

from __future__ import annotations

import gc
import os

import numpy as np
import torch

from rlinf.envs.world_model.world_model_dreamdojo_env import DreamDojoEnv

__all__ = ["DreamDojoStudentEnv"]


class DreamDojoStudentEnv(DreamDojoEnv):
    """Advance the causal DreamDojo student by one latent per policy chunk.

    Reset performs the student's native warmup with one initial image and 36
    demonstration actions. It retains the last nine RGB frames and eight model
    actions as causal context. Each later ``chunk_step`` consumes twelve 30 Hz
    policy actions, stride-samples four 10 Hz model actions, rebuilds the
    three-latent KV context, and generates four RGB frames.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.student_bootstrap_enabled = bool(
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

    def _validate_distilled_student_inputs(self) -> None:
        """Validate the full student DCP and its external Cosmos assets."""
        config_file = str(self.cfg.config_file)
        if "interactive/configs/" not in config_file:
            raise ValueError(
                "DreamDojo student requires an interactive config such as "
                "cosmos_predict2/_src/predict2/interactive/configs/"
                f"config_distill.py, got {config_file!r}."
            )

        experiment = str(self.cfg.experiment)
        if experiment == "dreamdojo_2b_1440_640_piper":
            raise ValueError(
                "dreamdojo_2b_1440_640_piper is the teacher experiment; the "
                "student requires an interactive self-forcing experiment."
            )

        checkpoint = str(self.cfg.dreamdojo_ckpt_path)
        if checkpoint.endswith(".pt") or checkpoint.rstrip("/").endswith("/model"):
            raise ValueError(
                "DreamDojo student expects the DCP checkpoint root, not a teacher "
                f".pt file or its model/ child: {checkpoint}"
            )
        if not checkpoint.startswith(("s3://", "msc://")):
            model_dir = os.path.join(os.path.expanduser(checkpoint), "model")
            metadata = os.path.join(model_dir, ".metadata")
            if not os.path.isfile(metadata):
                raise FileNotFoundError(
                    "DreamDojo student checkpoint metadata not found at "
                    f"{metadata}. Pass the DCP root directory."
                )
            self._validate_complete_dcp_model(model_dir)

        tokenizer_path = self.cfg.get("cosmos_tokenizer_path", None)
        if tokenizer_path in (None, "", "null"):
            raise ValueError(
                "DreamDojo student requires cosmos_tokenizer_path pointing to "
                "the full Cosmos-Predict2.5-2B tokenizer.pth file."
            )
        if not os.path.isfile(os.path.expanduser(str(tokenizer_path))):
            raise FileNotFoundError(
                f"DreamDojo student Cosmos tokenizer not found: {tokenizer_path}"
            )

        embedding_path = self.cfg.get(
            "cr1_embeddings_path",
            self.cfg.get("text_embedding_cache_path", None),
        )
        if embedding_path in (None, "", "null"):
            raise ValueError("DreamDojo student requires cr1_embeddings_path.")
        if not os.path.isfile(os.path.expanduser(str(embedding_path))):
            raise FileNotFoundError(
                f"DreamDojo student CR1 embedding cache not found: {embedding_path}"
            )

    @staticmethod
    def _validate_complete_dcp_model(model_dir: str) -> None:
        """Check that every shard range referenced by DCP metadata is present."""
        from torch.distributed.checkpoint import FileSystemReader

        try:
            metadata = FileSystemReader(model_dir).read_metadata()
        except Exception as exc:
            raise ValueError(
                f"Could not read DreamDojo student DCP metadata in {model_dir}."
            ) from exc

        required_sizes: dict[str, int] = {}
        for storage_info in metadata.storage_data.values():
            relative_path = storage_info.relative_path
            required_sizes[relative_path] = max(
                required_sizes.get(relative_path, 0),
                storage_info.offset + storage_info.length,
            )

        incomplete = []
        for relative_path, required_size in sorted(required_sizes.items()):
            shard_path = os.path.join(model_dir, relative_path)
            actual_size = (
                os.path.getsize(shard_path) if os.path.isfile(shard_path) else 0
            )
            if actual_size < required_size:
                incomplete.append(
                    f"{relative_path} ({actual_size}/{required_size} bytes)"
                )
        if incomplete:
            raise FileNotFoundError(
                "DreamDojo student DCP is incomplete; missing or truncated model "
                f"shards: {', '.join(incomplete)}"
            )

    def _student_checkpoint_experiment_opts(self) -> list[str]:
        """Build Hydra overrides for inference-only student construction."""
        tokenizer_path = os.path.expanduser(str(self.cfg.cosmos_tokenizer_path))
        experiment_opts = [
            "model.config.net_fake_score=null",
            # ``vae_pth`` is not declared in the registered tokenizer node, so
            # Hydra needs ``+`` to add the local checkpoint path.
            f"+model.config.tokenizer.vae_pth={tokenizer_path}",
        ]
        experiment_opts.extend(self.cfg.get("student_experiment_opts", []))
        return experiment_opts

    def _build_pipeline(self):
        """Load only the inference student from the self-forcing DCP."""
        self._ensure_dreamdojo_on_path()
        self._validate_distilled_student_inputs()

        if bool(self.cfg.get("dreamdojo_torch_compile", False)):
            raise ValueError(
                "DreamDojoStudentEnv currently requires "
                "dreamdojo_torch_compile=False while rebuilding its causal KV "
                "context for each environment."
            )

        from cosmos_predict2._src.predict2.interactive.inference import (
            action_video2world,
        )

        # The self-forcing training experiment defines another 2B fake-score
        # network. It is not used by inference, so disable it before model
        # construction instead of replicating it on every data-parallel GPU.
        experiment_opts = self._student_checkpoint_experiment_opts()
        original_loader = action_video2world.load_model_from_checkpoint

        def _load_student_checkpoint(*args, **kwargs):
            kwargs["experiment_opts"] = (
                list(kwargs.get("experiment_opts", [])) + experiment_opts
            )
            return original_loader(*args, **kwargs)

        action_video2world.load_model_from_checkpoint = _load_student_checkpoint
        try:
            pipe = action_video2world.ActionStreamingInference(
                config_path=self.cfg.config_file,
                experiment_name=self.cfg.experiment,
                ckpt_path=self.cfg.dreamdojo_ckpt_path,
                s3_credential_path=self.cfg.get(
                    "s3_credential_path", "credentials/s3_checkpoint.secret"
                ),
                cr1_embeddings_path=self.cfg.get(
                    "cr1_embeddings_path",
                    self.cfg.get("text_embedding_cache_path", None),
                ),
                context_parallel_size=int(
                    self.cfg.get("dreamdojo_context_parallel_size", 1)
                ),
                enable_fsdp=bool(self.cfg.get("dreamdojo_enable_fsdp", False)),
                torch_compile=False,
            )
        finally:
            action_video2world.load_model_from_checkpoint = original_loader

        # These are training-only networks. ``skip_teacher_init`` avoids loading
        # teacher weights but the experiment may still construct the modules.
        for attr in ("net_teacher", "net_fake_score", "net_discriminator_head"):
            if hasattr(pipe.model, attr):
                setattr(pipe.model, attr, None)

        # The 8-GPU preset keeps the cached CR1 embedding on each rank's GPU.
        # ActionStreamingInference names this attribute ``*_cpu`` but only ever
        # calls ``.to(model_device)`` on it, so replacing it with the resident
        # tensor also removes a roughly 98 MiB H2D copy from every chunk.
        model_device = pipe.model.tensor_kwargs["device"]
        model_dtype = pipe.model.tensor_kwargs["dtype"]
        self._student_text_embeddings_gpu = pipe.t5_text_embeddings_cpu.to(
            device=model_device, dtype=model_dtype
        )
        pipe.t5_text_embeddings_cpu = self._student_text_embeddings_gpu
        self._student_text_mask_gpu = torch.ones(
            (
                self._student_text_embeddings_gpu.shape[0],
                self._student_text_embeddings_gpu.shape[1],
            ),
            device=model_device,
            dtype=model_dtype,
        )
        gc.collect()
        self._clear_accelerator_cache()
        return pipe

    def _validate_student_temporal_contract(self) -> None:
        """Check the environment cadence against the loaded student network."""
        if self.student_actions_per_latent != 4:
            raise ValueError(
                "The current DreamDojo student requires student_actions_per_latent=4."
            )
        if self.student_cache_latents != 3:
            raise ValueError(
                "The current DreamDojo student requires student_cache_latents=3."
            )
        if self.gen_frames != self.student_actions_per_latent:
            raise ValueError(
                "Student chunk_step must generate one latent: chunk // "
                f"action_stride must be 4, got {self.chunk} // "
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
                "Student warmup must provide exactly 12 model actions, i.e. 36 "
                "policy actions when action_stride=3."
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
        """Reset and build the student's native causal warmup prefix."""
        obs, info = super().reset(*args, **kwargs)
        self.student_bootstrap_complete = False
        if self.student_bootstrap_enabled:
            self._bootstrap_student_context()
            self.student_bootstrap_complete = True
            return self._wrap_obs(), info

        # Explicit fallback for synthetic callers without demonstration data.
        initial = self.current_obs.detach().permute(0, 3, 1, 2)
        self._student_condition_frames = initial.unsqueeze(2).repeat(
            1, 1, self.student_context_pixel_frames, 1, 1
        )
        self._student_action_history = torch.zeros(
            self.num_envs,
            self.student_history_actions,
            self.model_action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        return obs, info

    def _load_student_bootstrap_policy_actions(self) -> torch.Tensor:
        """Load the 36-action, 30 Hz demonstration prefix for every env."""
        episode_indices = self._last_reset_episode_indices
        if episode_indices is None or len(episode_indices) != self.num_envs:
            raise RuntimeError(
                "DreamDojo student did not receive reset episode indices for warmup."
            )

        actions_per_env = []
        for env_idx, episode_idx in enumerate(episode_indices):
            trajectory = self.dataset._load_trajectory(int(episode_idx))
            if len(trajectory) < self.student_bootstrap_policy_actions:
                raise ValueError(
                    "DreamDojo student warmup needs at least "
                    f"{self.student_bootstrap_policy_actions} frames/actions in "
                    f"{self.dataset.npy_files[int(episode_idx)]}, but found "
                    f"{len(trajectory)}. Re-export reset data with "
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
        """Generate one conditioning frame plus twelve student warmup frames."""
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
                "DreamDojo student warmup returned invalid video shape "
                f"{tuple(video.shape)}."
            )
        expected_frames = self.student_bootstrap_model_actions + 1
        if video.shape[2] < expected_frames:
            raise RuntimeError(
                "DreamDojo student warmup generated too few frames: "
                f"got {video.shape[2]}, need {expected_frames}."
            )
        return video[:, :, -expected_frames:].detach().cpu()

    @torch.no_grad()
    def _bootstrap_student_context(self) -> None:
        """Warm up and retain the native nine-frame/eight-action context."""
        policy_actions = self._load_student_bootstrap_policy_actions()
        model_actions = policy_actions[:, :: self.action_stride]
        if model_actions.shape[1] != self.student_bootstrap_model_actions:
            raise RuntimeError(
                "Student warmup stride produced "
                f"{model_actions.shape[1]} actions; expected "
                f"{self.student_bootstrap_model_actions}."
            )
        if self.policy_action_format == "absolute":
            model_actions = self._absolute_to_dreamdojo_delta(model_actions)

        condition_frames = []
        action_history = []
        final_frames = []
        for env_idx in range(self.num_envs):
            cosmos_actions = self._build_model_action(model_actions[env_idx]).to(
                self.device
            )
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

        self._student_condition_frames = torch.stack(condition_frames, dim=0).to(
            self.device
        )
        self._student_action_history = torch.stack(action_history, dim=0).to(
            self.device
        )
        self.current_obs = torch.stack(final_frames, dim=0).to(self.device)
        self.current_states = policy_actions[:, -1].to(self.device)
        self._last_action_state = policy_actions[:, -1].clone()
        # Warmup frames are context only and never enter reward or GRPO loss.
        self.last_chunk_frames = None

    def _build_student_condition(self, env_idx: int, current_actions: torch.Tensor):
        """Encode rolling pixels and construct prefill/generation conditions."""
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
            actions_np=all_actions.detach().cpu().numpy(),
            fps=self.student_condition_fps,
            num_latent_conditional_frames=0,
        )

        model = self.pipe.model
        model._normalize_video_databatch_inplace(data_batch)
        model._augment_image_dim_inplace(data_batch)

        data_batch["t5_text_embeddings"] = self._student_text_embeddings_gpu
        data_batch["t5_text_mask"] = self._student_text_mask_gpu
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
        """Generate one causal latent and decode its four new RGB frames."""
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
                f"got {tuple(x0.shape)}."
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

        # The 8-GPU preset keeps both DiT and VAE resident. Do not move the DiT
        # to CPU between generation and decode as the single-5090 path does.
        decoded = self.pipe._decode(decode_latents).clip(min=-1, max=1)
        if decoded.shape[2] < self.student_actions_per_latent:
            raise RuntimeError(
                "DreamDojo student decoded fewer than four frames: "
                f"{tuple(decoded.shape)}"
            )
        return decoded[:, :, -self.student_actions_per_latent :]

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions) -> None:
        """Consume twelve policy actions and advance one latent per env."""
        if actions.shape[:2] != (self.num_envs, self.chunk):
            raise ValueError(
                "DreamDojo student expected action shape "
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
            model_actions = self._build_model_action(actions[env_idx]).to(self.device)
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

        self.current_obs = torch.stack(new_last_frames, dim=0).to(self.device)
        self.last_chunk_frames = torch.stack(new_chunks, dim=0).to(self.device)
