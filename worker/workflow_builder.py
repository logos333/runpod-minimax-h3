from __future__ import annotations

from typing import Any

from worker.config import WorkerSettings
from worker.schemas import GenerationMode, GenerationRequest


class MiniMaxH3WorkflowBuilder:
    """Builds verified ComfyUI API graphs for MATLOWAI MiniMax-H3 Fused Turbo."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    def build(
        self,
        request: GenerationRequest,
        ref_image_names: list[str],
        ref_audio_names: list[str],
        first_frame_name: str | None,
        last_frame_name: str | None,
        guide_audio_name: str | None,
        output_prefix: str,
    ) -> tuple[dict[str, Any], str]:
        """Return (comfy_prompt_graph, save_video_node_id)."""
        graph: dict[str, Any] = {}
        num_frames = request.aligned_frame_count()

        # 1. Core Model Loaders (MATLOWAI single fused checkpoint + Comfy-Org Qwen3-VL & VAEs)
        graph["10"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": self._settings.unet_filename,
                "weight_dtype": "default",
            },
        }
        graph["11"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": self._settings.text_encoder_filename,
                "type": "minimax",
                "device": "default",
            },
        }
        graph["12"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": self._settings.video_vae_filename},
        }
        graph["13"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": self._settings.audio_vae_filename},
        }

        # 2. Model patch chain: UNET -> [optional SLA] -> [optional FFN chunker] -> SigmaShift
        model_ref: list[Any] = ["10", 0]

        if request.enable_sla:
            graph["20"] = {
                "class_type": "H3SLAAttention",
                "inputs": {
                    "model": model_ref,
                    "sparsity_ratio": request.sla_sparsity,
                    "block_size": self._settings.default_sla_block_size,
                    "min_seq_len": self._settings.default_sla_min_seq_len,
                    "dense_last_steps": 0,
                    "protect_audio": True,
                    "enabled": True,
                },
            }
            model_ref = ["20", 0]

        if request.low_vram_chunk_ffn or self._settings.default_low_vram_chunk_ffn:
            graph["21"] = {
                "class_type": "MiniMaxChunkFeedForward",
                "inputs": {
                    "model": model_ref,
                    "chunks": 4,
                    "seq_threshold": 4096,
                },
            }
            model_ref = ["21", 0]

        graph["22"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": model_ref,
                "shift_video": request.shift_video,
                "shift_audio": request.shift_audio,
            },
        }
        patched_model_ref: list[Any] = ["22", 0]

        # 3. Task conditioning & latent creation
        if request.mode in (GenerationMode.REF2VA, GenerationMode.HYBRID):
            ref_inputs: dict[str, Any] = {
                "clip": ["11", 0],
                "vae": ["12", 0],
                "audio_vae": ["13", 0],
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "length": num_frames,
                "ref_image_size": request.ref_image_size,
            }

            for idx, img_name in enumerate(ref_image_names):
                node_id = str(100 + idx)
                graph[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": img_name},
                }
                ref_inputs[f"ref_images.ref_image_{idx}"] = [node_id, 0]

            for idx, aud_name in enumerate(ref_audio_names):
                node_id = str(150 + idx)
                graph[node_id] = {
                    "class_type": "LoadAudio",
                    "inputs": {"audio": aud_name},
                }
                ref_inputs[f"ref_audios.ref_audio_{idx}"] = [node_id, 0]

            graph["30"] = {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": ref_inputs,
            }
            cond_ref: list[Any] = ["30", 0]
            latent_ref: list[Any] = ["30", 1]

            # Hybrid mode: anchor explicit first and/or last frame on top of Ref2VA conditioning
            if request.mode == GenerationMode.HYBRID:
                if first_frame_name:
                    graph["180"] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": first_frame_name},
                    }
                    graph["31"] = {
                        "class_type": "MiniMaxH3AddGuide",
                        "inputs": {
                            "positive": cond_ref,
                            "vae": ["12", 0],
                            "latent": latent_ref,
                            "image": ["180", 0],
                            "frame_idx": 0,
                        },
                    }
                    cond_ref = ["31", 0]

                if last_frame_name:
                    graph["181"] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": last_frame_name},
                    }
                    graph["32"] = {
                        "class_type": "MiniMaxH3AddGuide",
                        "inputs": {
                            "positive": cond_ref,
                            "vae": ["12", 0],
                            "latent": latent_ref,
                            "image": ["181", 0],
                            "frame_idx": -1,
                        },
                    }
                    cond_ref = ["32", 0]
        else:
            # FL2VA / T2VA mode
            i2v_inputs: dict[str, Any] = {
                "clip": ["11", 0],
                "vae": ["12", 0],
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "length": num_frames,
            }
            if first_frame_name:
                graph["180"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": first_frame_name},
                }
                i2v_inputs["first_frame"] = ["180", 0]
            if last_frame_name:
                graph["181"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": last_frame_name},
                }
                i2v_inputs["last_frame"] = ["181", 0]

            graph["30"] = {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": i2v_inputs,
            }
            cond_ref = ["30", 0]
            latent_ref = ["30", 1]

        # 3b. Hard Audio-to-Video (A2V) Inpaint Lock: freeze target audio stream (noise_mask=0.0)
        # via VAEEncodeAudio -> SolidMask(0.0) -> SetLatentNoiseMask -> LTXVConcatAVLatent
        if guide_audio_name:
            audio_t = max(1, round((num_frames / 24.0) * 40))
            graph["190"] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": guide_audio_name},
            }
            graph["191"] = {
                "class_type": "VAEEncodeAudio",
                "inputs": {
                    "audio": ["190", 0],
                    "vae": ["13", 0],
                },
            }
            graph["192"] = {
                "class_type": "SolidMask",
                "inputs": {
                    "value": 0.0,
                    "width": audio_t,
                    "height": 2,
                },
            }
            graph["193"] = {
                "class_type": "SetLatentNoiseMask",
                "inputs": {
                    "samples": ["191", 0],
                    "mask": ["192", 0],
                },
            }
            graph["194"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": latent_ref,
                    "audio_latent": ["193", 0],
                },
            }
            latent_ref = ["194", 0]

        # 4. Sampling pipeline (res_multistep / simple / 4-step Turbo)
        graph["40"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": request.seed},
        }
        graph["41"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": self._settings.default_sampler},
        }
        graph["42"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": patched_model_ref,
                "scheduler": self._settings.default_scheduler,
                "steps": request.steps,
                "denoise": 1.0,
            },
        }
        graph["43"] = {
            "class_type": "BasicGuider",
            "inputs": {
                "model": patched_model_ref,
                "conditioning": cond_ref,
            },
        }
        graph["44"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["40", 0],
                "guider": ["43", 0],
                "sampler": ["41", 0],
                "sigmas": ["42", 0],
                "latent_image": latent_ref,
            },
        }

        # 5. Joint Video + Audio Decode & Mux
        graph["50"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["44", 0], "vae": ["12", 0]},
        }
        graph["51"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["44", 0], "vae": ["13", 0]},
        }
        graph["52"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["50", 0],
                "audio": ["51", 0],
                "fps": request.fps,
                "bit_depth": 8,
            },
        }
        save_node_id = "60"
        graph[save_node_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["52", 0],
                "format": "mp4",
                "codec": "h264",
                "filename_prefix": output_prefix,
            },
        }
        return graph, save_node_id
