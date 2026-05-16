import os
import tempfile
import cv2
import gradio as gr
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

print("Patchcore is loaded...")
model = Patchcore()
engine = Engine()
checkpoint_path = "model/weights/lightning/model.ckpt"


def predict_defect(input_img):
    if input_img is None:
        return None, "Image not loaded", 0.0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_file.name, img_bgr)
        temp_path = temp_file.name

    try:
        dataset = PredictDataset(
            path=temp_path,
            image_size=(256, 256)
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=checkpoint_path
        )

        for prediction in predictions:
            output_visual = input_img

            if hasattr(prediction, "heat_map") and getattr(prediction, "heat_map") is not None:
                heatmap_tensor = getattr(prediction, "heat_map")
                if hasattr(heatmap_tensor, "cpu"):
                    heatmap_tensor = heatmap_tensor.cpu().numpy()
                output_visual = cv2.cvtColor(heatmap_tensor, cv2.COLOR_BGR2RGB)

            elif hasattr(prediction, "anomaly_map") and getattr(prediction, "anomaly_map") is not None:
                amap = getattr(prediction, "anomaly_map")

                if hasattr(amap, "cpu"):
                    amap = amap.cpu().numpy()

                amap = amap.squeeze()

                amap_normalized = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

                heatmap = cv2.applyColorMap(amap_normalized, cv2.COLORMAP_JET)
                output_visual = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            pred_label = "Anomaly (Defect)" if getattr(prediction, "pred_label", 0) == 1 else "Norm (Normal)"
            pred_score = float(getattr(prediction, "pred_score", 0.0))

            return output_visual, pred_label, round(pred_score, 4)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return None, "Processing error", 0.0


with gr.Blocks(title="Defect Detection") as demo:
    gr.Markdown("# Defect detection based on Anomalib (Patchcore)")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="numpy")
            btn_submit = gr.Button("Run analysis", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Anomaly Map", type="numpy")
            output_label = gr.Textbox(label="Model verdict")
            output_score = gr.Number(label="Score")

    btn_submit.click(
        fn=predict_defect,
        inputs=[input_image],
        outputs=[output_image, output_label, output_score]
    )

if __name__ == "__main__":
    demo.launch()