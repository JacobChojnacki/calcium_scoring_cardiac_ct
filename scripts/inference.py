import logging

from pathlib import Path

import fire
import SimpleITK as sitk
import torch

from agatston.models.RaUNet2D import RaUNet2D
from agatston.models.RaUNet3D import RaUNet3D
from agatston.models.UNet2D import UNet2D
from agatston.models.UNet3D import UNet3D
from agatston.models.UNETR2D import UNETR2D
from agatston.tools import agatston_tool
from monai.inferers import SliceInferer, sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    SaveImage,
    ScaleIntensityRange,
    Spacing,
    ToTensor,
)


class Inference:
    def __init__(
        self,
        output_dir: str = './',
        model: str = 'raunet',
        model_ckpt: str = '',
        type_of_input_data: str = '2D',
        only_plaques: bool = True,
        soft_inference: bool = False,
    ):
        self._output_dir = output_dir
        self._model_name = model
        self._model_ckpt = model_ckpt
        self._type_of_input_data = type_of_input_data
        self._only_plaques = only_plaques
        self._soft_inference = soft_inference
        self._image_spacing = [0.3, 0.3, 0.3]
        self._writer = 'NibabelWriter'
        self._output_ext = '.nii.gz'
        self.transformations = self._create_transformations()
        self.model = self._load_model()

    def _create_transformations(self):
        common_transforms = [
            LoadImage(),
            EnsureType(),
            EnsureChannelFirst(),
            Orientation(axcodes='RAS'),
            SaveImage(
                output_postfix='',
                output_dir=self._output_dir,
                output_ext=self._output_ext,
                writer=self._writer,
                resample=False,
                print_log=True,
            ),
            ScaleIntensityRange(a_min=-200, a_max=1300, b_min=0, b_max=1, clip=True),
            ToTensor(),
        ]

        if self._type_of_input_data == '3D':
            common_transforms.insert(3, Spacing(pixdim=self._image_spacing, mode='bilinear'))

        return Compose(common_transforms)

    def _load_model(self):
        model_mapping = {
            'raunet2d': RaUNet2D,
            'raunet3d': RaUNet3D,
            'unet2d': UNet2D,
            'unet3d': UNet3D,
            'unetr2d': UNETR2D,
        }

        model_key = f'{self._model_name}{self._type_of_input_data.lower()}'
        model_class = model_mapping.get(model_key)

        if model_class is None:
            raise ValueError(f'Unsupported model: {self._model_name} for input type: {self._type_of_input_data}')

        model = model_class()
        checkpoint = torch.load(self._model_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()
        return model

    def _run_inference(self, loaded_input):
        with torch.no_grad():
            if self._type_of_input_data == '2D':
                inferer = SliceInferer(
                    roi_size=[512, 512], sw_batch_size=1, spatial_dim=2, cval=-1, device='cuda', progress=True
                )
                output = inferer(loaded_input, self.model.forward)
            else:
                output = sliding_window_inference(
                    loaded_input,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    cval=-1,
                    progress=True,
                    device='cuda',
                    predictor=self.model.forward,
                )
        output = torch.squeeze(output)

        if self._soft_inference:
            inferences_channel = {}
            for idx, inference_channel in enumerate(output):
                inferences_channel[idx] = inference_channel
            return inferences_channel
        else:
            output = output.argmax(0)
            if self._only_plaques:
                output[output > 1] = 0
            return output

    def _count_plaques(self, inference_model: sitk.Image):
        cc = sitk.ConnectedComponentImageFilter()
        cc.Execute(inference_model)
        return cc.GetObjectCount()

    def inference(self, image_path):
        try:
            loaded_input = self.transformations(image_path)
            loaded_input = loaded_input.unsqueeze(0).to('cuda')

            output = self._run_inference(loaded_input)

            if self._soft_inference:
                for infer in range(len(output)):
                    save = SaveImage(output_dir=self._output_dir, output_postfix=f'inference_{infer}')
                    save(output[infer])
            else:
                save_image = SaveImage(
                    output_dir=self._output_dir, output_postfix='inference', output_ext=self._output_ext
                )
                save_image(output)

            return output
        except Exception as e:
            print(f'Error during inference: {e}')
            raise

    def calculate_agatston(self, image_file_path, inference_file_path):
        try:
            # Convert paths to Path objects
            image_path = Path(image_file_path)
            inference_path = Path(inference_file_path)
            output_dir_path = Path(self.output_dir)

            # Ensure the output directory exists
            output_dir_path.mkdir(parents=True, exist_ok=True)

            # Calculate Agatston score and plaque volume
            agatston_score = agatston_tool.agatston_score(image_path, inference_path, is_calcium_volume=False)
            plaque_volume = agatston_tool.agatston_score(image_path, inference_path, is_calcium_volume=True)

            stem_name = str(image_path.stem).replace('.nii', '')
            # Prepare output file path
            output_file_path = output_dir_path / f'{stem_name}_report.txt'

            # Write results to the output file
            with open(output_file_path, 'w') as f:
                f.write(
                    f'Results for {image_path.stem}\n'
                    f'Agatston Score: {round(agatston_score, 2)}\n'
                    f'Plaque Volume: {round(plaque_volume, 2)} mm^3\n'
                )
                f.close()
            logging.info(f'Report generated: {output_file_path}')

        except Exception as e:
            logging.error(f'An error occurred while calculating the Agatston score: {e}')


if __name__ == '__main__':
    fire.Fire(Inference)
