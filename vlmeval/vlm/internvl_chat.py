import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import torch.distributed as dist
import torchvision.transforms as T
import transformers

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_local_rank_and_local_world_size():
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        return 0, 1

    if 'SLURM_NTASKS_PER_NODE' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_world_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
        local_rank = rank % local_world_size

        assert rank % local_world_size == 0
        return local_rank, local_world_size

    if 'LOCAL_RANK' in os.environ and 'LOCAL_WORLD_SIZE' in os.environ:
        return int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])

    raise NotImplementedError(
        "Fail to get local_rank and local_world_size! "
        "Please ensure that you set the environment variable "
        "`LOCAL_RANK` and `LOCAL_WORLD_SIZE`"
    )


def split_model(model_path):
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    local_rank, local_world_size = get_local_rank_and_local_world_size()
    visible_devices = [i for i in range(local_rank, num_gpus, local_world_size)]

    print(f'[Rank {rank}] {local_rank=} {local_world_size=}')

    if len(visible_devices) > 1:
        device_map = {}
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        num_gpus_for_vit = 1
        num_gpus_for_llm = len(visible_devices) - num_gpus_for_vit

        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_llm + 1
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus_for_llm
        num_layers_per_gpu[0] -= 4

        while sum(num_layers_per_gpu) < num_layers:
            for i in range(1, len(num_layers_per_gpu)-1):
                num_layers_per_gpu[i] += 1

        if rank == 0:
            print(f'{num_layers_per_gpu=}, {num_layers=}')

        device_idx = num_gpus_for_vit
        device_cnt = 0
        for i in range(num_layers):
            if device_cnt >= num_layers_per_gpu[device_idx-num_gpus_for_vit]:
                device_idx += 1
                device_cnt = 0

            device_cnt += 1
            device_map[f'language_model.model.layers.{i}'] = visible_devices[device_idx]

        num_layers = config.vision_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_vit
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu, num_gpus_for_vit - 1)
            device_map[f'vision_model.encoder.layers.{i}'] = visible_devices[device_idx]

        device_map['vision_model.embeddings'] = visible_devices[0]
        device_map['mlp1'] = visible_devices[num_gpus_for_vit-1]
        # InternLM2
        device_map['language_model.model.tok_embeddings'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.output'] = visible_devices[-1]
        # Qwen2
        device_map['language_model.model.embed_tokens'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.lm_head'] = visible_devices[-1]

    else:
        device_map = {'': visible_devices[0]}

    print(f'[Rank {rank}] {device_map=}')

    return device_map, visible_devices


class InternVLChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='OpenGVLab/InternVL-Chat-V1-5', version='V1.0', **kwargs):
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device_map, visible_devices = split_model(model_path=model_path)

        self.device = visible_devices[0]
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               device_map=device_map).eval()

        self.image_size = self.model.config.vision_config.image_size
        self.version = version
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if self.version == 'V1.1':
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=5)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        self.kwargs = kwargs_default

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        elif dataset is not None and listinstr(['MMBench-Video'], dataset):
            self.max_num = 1
        else:
            self.max_num = 6

    def generate_v2(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<image-{image_idx}>'
                    image_idx += 1
            prompt = ' '.join([f'<image-{i + 1}>: <image>' for i in range(image_num)]) + '\n' + prompt
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('<image-1><image-2><image-3><image-4><image-5><image-6><image-7><image-8>', '')
            prompt = prompt.replace('<image-9><image-10><image-11><image-12><image-13><image-14><image-15><image-16>', '')
            for i in range(8):
                prompt = prompt.replace(f'<image-{i + 1}>', f'Frame{i+1}')
            prompt = prompt.replace('\nAnswer:', '')
            prompt += '\nAnswer the question using a single word or phrase.'
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                curr_pixel_values = load_image(file_name, max_num=self.max_num, upscale=image_idx == 0 and listinstr(['MMMU_DEV_VAL'], dataset)).cuda().to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = load_image(image_path, max_num=self.max_num, upscale=listinstr(['MMMU_DEV_VAL'], dataset)).cuda().to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        with torch.no_grad():
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=prompt,
                    generation_config=self.kwargs,
                    verbose=True
                )
            except torch.cuda.OutOfMemoryError:
                response = 'A'
                torch.cuda.empty_cache()
        return response

    def generate_v1_5(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            pixel_values_list = []
            for file_name in image_path:
                pixel_values_list.append(load_image(file_name, max_num=self.max_num).cuda().to(torch.bfloat16))
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            pixel_values = load_image(image_path, max_num=self.max_num).cuda().to(torch.bfloat16)
        else:
            pixel_values = None
            
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                       question=prompt, generation_config=self.kwargs)
        return response

    def generate_v1_2(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values,
                                       question=prompt, generation_config=self.kwargs)
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        if self.version in ['V1.2', 'V1.5']:
            return self.generate_v1_2(message, dataset)
        if self.version == 'V1.5':
            return self.generate_v1_5(message, dataset)
        elif self.version == 'V2.0':
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')