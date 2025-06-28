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

plt.imshow(annotated_frame)
plt.axis('off')
plt.show()