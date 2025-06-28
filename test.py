query = input()

from groundingdino.util.inference import load_model, load_image, predict, annotate
import matplotlib.pyplot as plt

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

config_path = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weight_path = "../GroundingDINO/groundingdino_swint_ogc.pth"

model = load_model(config_path, weight_path)

image_source, image = load_image("./pre.png")

boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=query,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

print(image.size())
"""
boxes_px = boxes.copy()
boxes_px[:, 0] *= W  # x_center
boxes_px[:, 1] *= H  # y_center
boxes_px[:, 2] *= W  # width
boxes_px[:, 3] *= H  # height

x0 = boxes_px[:, 0] - boxes_px[:, 2] / 2
y0 = boxes_px[:, 1] - boxes_px[:, 3] / 2
x1 = boxes_px[:, 0] + boxes_px[:, 2] / 2
y1 = boxes_px[:, 1] + boxes_px[:, 3] / 2
"""
print(f"{x0} {y0} {x1} {y1}")
print("done")
plt.imshow(annotated_frame)
plt.axis('off')
plt.savefig("output.png", bbox_inches='tight')