import PIL
import torch
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_segmentation_masks, save_image
import logging
from PIL import Image
import warnings
from segnet_class import SegNet

warnings.filterwarnings("ignore")

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

DEVICE = "cpu"


def pretrained_segnet(params_path: str, device: str):
    """load model and weights"""
    model = SegNet().to(device)
    model.load_state_dict(torch.load(params_path, map_location=torch.device(device)))
    return model


def segment(path: str, inp_size: int):
    """detecting function"""
    try:
        image = Image.open(path).convert('RGB')
    except PIL.UnidentifiedImageError:
        m_logger.error(f'something wrong with image')
        status = 'Fail'
        seg_result = 'no result'
        return status, path, seg_result
    shape = (image.size[1], image.size[0])
    if shape != (inp_size, inp_size):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((inp_size, inp_size))
        ])
    else:
        transformer = transforms.ToTensor(),
    image = transformer(image)
    model = pretrained_segnet('segnet_bones.pth', DEVICE)
    m_logger.info(f'model loaded')
    with torch.no_grad():
        x = image.to(DEVICE).unsqueeze(0)
        predictions = model.eval()(x)
    mask = predictions[0]
    mask_dense = torch.sum(torch.sigmoid(mask))
    mask = torch.round(torch.sigmoid(mask)).to(torch.bool)
    if mask_dense.detach().numpy() < .54 * inp_size ** 2:  # experimental value: mask shouldn't be very big
        seg_result = 'bones found'
    else:
        seg_result = 'no bones'
    m_logger.info(f'{seg_result}')
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    if seg_result == 'bones found':
        output_image = draw_segmentation_masks(image, mask, alpha=0.5, colors="blue").cpu().detach()
        m_logger.info(f'segmentation completed')
    else:
        output_image = image.cpu().detach()
    output_image = output_image.unsqueeze(0).permute(0, 1, 2, 3) / 255
    img_path = path.split('/')[0] + '/res_' + path.split('/')[1]
    save_image(output_image, img_path)
    m_logger.info(f'image saved')
    status = 'OK'
    return status, img_path, seg_result
