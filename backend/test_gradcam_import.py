# test_gradcam_import.py
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    print("GradCAM and show_cam_on_image imported successfully")
except Exception as e:
    print("Import failed:", type(e).__name__, e)
