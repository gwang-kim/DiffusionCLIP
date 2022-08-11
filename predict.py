# prepare the weights and face-alignment
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bunzip2 shape_predictor_68_face_landmarks.dat.bz2

import os
import torch
import shutil
import tempfile

import argparse
import yaml
from PIL import Image
import warnings

warnings.filterwarnings(action="ignore")
from cog import BasePredictor, Path, Input

from diffusionclip import DiffusionCLIP
from main import dict2namespace


class Predictor(BasePredictor):
    def setup(self):
        self.configs = {
            "ImageNet style transfer": "imagenet.yml",
            "Human face manipulation": "celeba.yml",
            "Dog face manipulation": "afhq.yml",
            "Tennis ball manipulation": "imagenet.yml",
        }

        self.model_paths = {
            "ImageNet style transfer": {
                "Watercolor art": "imagenet_watercolor_t601.pth",
                "Pointillism art": "imagenet_pointillism_t601.pth",
                "Painting by Gogh": "imagenet_gogh_t601.pth",
                "Cubism art": "imagenet_cubism_t601.pth",
            },
            "Human face manipulation": {
                "Pixar": "human_pixar_t601.pth",
                "Neanderthal": "human_neanderthal_t601.pth",
                "Painting by Gogh": "human_gogh_t601.pth",
                "Tanned": "human_tanned_t201.pth",
                "Female → Male": "human_male_t401.pth",
                "Sketch": "human_sketch_t601.pth",
                "With makeup": "human_with_makeup_t301.pth",
                "Without makeup": "human_without_makeup_t301.pth",
            },
            "Dog face manipulation": {
                "Bear": "dog_bear_t500.pth",
                "Hamster": "dog_hamster_t601.pth",
                "Yorkshire Terrier": "dog_yorkshire_t601.pth",
                "Nicolas Cage": "dog_nicolas_t601.pth",
                "Zombie": "dog_zombie_t601.pth",
                "Venom": "dog_venome_t601.pth",
                "Painting by Gogh": "dog_gogh_t500.pth",
            },
        }

    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        manipulation: str = Input(
            default="ImageNet style transfer",
            choices=[
                "ImageNet style transfer",
                "Human face manipulation",
                "Dog face manipulation",
            ],
            description="Choose a manipulation type."
            #           " Human face manipulation expects aligned image, pre-process with
            #"https://replicate.com/cjwbw/face-align-cog for images that are not aligned.",
        ),
        edit_type: str = Input(
            default="ImageNet Style Transfer - Watercolor art",
            choices=[
                "ImageNet style transfer - Watercolor art",
                "ImageNet style transfer - Pointillism art",
                "ImageNet style transfer - Painting by Gogh",
                "ImageNet style transfer - Cubism art",
                "Human face manipulation - Pixar",
                "Human face manipulation - Neanderthal",
                "Human face manipulation - Sketch",
                "Human face manipulation - Painting by Gogh",
                "Human face manipulation - Tanned",
                "Human face manipulation - With makeup",
                "Human face manipulation - Without makeup",
                "Human face manipulation - Female → Male",
                "Dog face manipulation - Bear",
                "Dog face manipulation - Hamster",
                "Dog face manipulation - Yorkshire Terrier",
                "Dog face manipulation - Nicolas Cage",
                "Dog face manipulation - Zombie",
                "Dog face manipulation - Venom",
                "Dog face manipulation - Painting by Gogh",
            ],
            description="Choose corresponding edit type available for model chosen.",
        ),
        degree_of_change: float = Input(
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        n_test_step: int = Input(
            default=12,
            ge=5,
            le=100,
        ),
    ) -> Path:
        # sanity check
        assert edit_type.startswith(
            manipulation
        ), f"Please choose the available edit types for {manipulation}."

        edit_type = edit_type.split("- ")[-1]
        model_path = os.path.join(
            "checkpoint", self.model_paths[manipulation][edit_type]
        )
        t_0 = int(model_path.split("_t")[-1].replace(".pth", ""))
        exp_dir = "temp_dir"
        os.makedirs(exp_dir, exist_ok=True)

        # Test arg, config
        align_face = 1 if manipulation == "Human face manipulation" else 0
        n_inv_step = 40
        args_dic = {
            "config": self.configs[manipulation],
            "t_0": t_0,
            "n_inv_step": int(n_inv_step),
            "n_test_step": int(n_test_step),
            "sample_type": "ddim",
            "eta": 0.0,
            "bs_test": 1,
            "model_path": model_path,
            "img_path": str(image),
            "deterministic_inv": 1,
            "hybrid_noise": 0,
            "n_iter": 1,
            "align_face": align_face,
            "image_folder": exp_dir,
            "model_ratio": degree_of_change,
            "edit_attr": None,
            "src_txts": None,
            "trg_txts": None,
        }
        args = dict2namespace(args_dic)

        with open(os.path.join("configs", args.config), "r") as f:
            config_dic = yaml.safe_load(f)
        config = dict2namespace(config_dic)
        config.device = "cuda:0"

        # Edit
        runner = DiffusionCLIP(args, config)
        runner.edit_one_image()

        out_image = Image.open(
            f"{exp_dir}/3_gen_t{t_0}_it0_ninv{n_inv_step}_ngen{n_test_step}_mrat{degree_of_change}_{model_path.split('/')[-1].replace('.pth', '')}.png"
        )
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        out_image.save(str(out_path))
        shutil.rmtree(exp_dir)

        return out_path
