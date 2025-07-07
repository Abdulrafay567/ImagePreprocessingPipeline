from fastapi import FastAPI, UploadFile, File, HTTPException,Response,Form
from typing import List, Optional,Tuple
import numpy as np
import json
import cv2
from Image_processing import get_image_metadata, grayscalee,binarize,resize,normalize_image,remove_noisee,detect_and_denoise,hybrid_deblur,enhance_contrast_with_clahe,auto_gamma_correction,hard_crop,detect_and_straighten_by_moments,crop_with_paddleocr
from datetime import datetime
import os
app = FastAPI()
# Helper function to save the original image
def _save_original_image(image: np.ndarray, output_dir: str = "original_images") -> str:
    """Saves the original image with a timestamp and returns its path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(output_dir, f"original_{timestamp}.png")
    cv2.imwrite(save_path, image)
    return save_path
@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    actions: str = "metadata",
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    keep_aspect: Optional[bool] = True,
    grayscale_threshold: Optional[int] = None,
    interpolation: Optional[str] = "bilinear",
    # normalize: Optional[bool] = False,
    # remove_noise: Optional[bool] = False,
    # grayscale: Optional[bool] = False,
    crop_points: Optional[List[str]] = Form(None)     
):
    try:
        actions_list = [action.strip().lower() for action in actions.split(",")]
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        results = {}
        
        # Save the original image
        original_path = _save_original_image(image)
        results["original_saved_path"] = original_path

        # Work on a copy for processing
        processed_image = image.copy()
        if "metadata" in actions_list:
            results["metadata"] = get_image_metadata(processed_image)
        if "grayscale" in actions_list:
            processed_image = grayscalee(processed_image, threshold_value=grayscale_threshold)
            

        # Preprocess before binarization
        if "binarize" in actions_list:
            processed_image = binarize(processed_image)  
        if "normalize" in actions_list:
            processed_image =  normalize_image(processed_image)         
        # Main actions processing
        if "resize" in actions_list:
            processed_image = resize(
                processed_image,width=resize_width,height=resize_height,keep_aspect=keep_aspect,interpolation=interpolation)
        if "remove_noise" in actions_list:  # Standalone uses smart detection
            processed_image = detect_and_denoise(processed_image)
        if "deblur" in actions_list:
            processed_image = hybrid_deblur(processed_image)    
        
        if "contrast correction" in actions_list:
            processed_image = enhance_contrast_with_clahe(processed_image)
         # 🔁 Ensure image has 3 color channels before returning
        if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        if "adjust_brightness" in actions_list:
            processed_image = auto_gamma_correction(processed_image)
        if "crop" in actions_list:
            try:
                crop_coords = [int(p.strip()) for p in crop_points[0].split(",")]
                if len(crop_coords) != 4:
                    raise ValueError("Invalid crop point format")

                x1, y1, x2, y2 = crop_coords
                processed_image = hard_crop(processed_image, (x1, y1), (x2, y2))

            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"400: crop_points must be 4 comma-separated integers (x1,y1,x2,y2) — Error: {str(e)}"
                )
        if "deskew" in actions_list:
            processed_image = detect_and_straighten_by_moments(processed_image)
            
        if "east_crop" in actions_list:
            processed_image = crop_with_paddleocr(processed_image,padding=20, det_only=True)              
        # Encode and return
        _, buffer = cv2.imencode(".png", processed_image)
        return Response(
            content=buffer.tobytes(),
            media_type="image/png",
            headers={"X-Image-Metadata": json.dumps(results)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
