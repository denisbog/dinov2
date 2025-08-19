from PIL import Image
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
import torch


def prepare_img():
    im = Image.open("images/room.jpg")
    return im


def check():
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    model = DepthAnythingForDepthEstimation.from_pretrained(model_name)

    weight = model.state_dict()[
        "backbone.encoder.layer.0.attention.attention.key.weight"
    ]
    print(weight.dtype)
    print(weight.shape)
    print(weight[0][0:10])

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image = prepare_img()
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    depth.show()


if __name__ == "__main__":
    check()
