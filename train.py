import albumentations as A
from ultralytics import YOLO
from ultralytics.data import augment

# --- 1. DEFINE THE "SPICY PIXEL" AUGMENTATION ---
# This overrides the default YOLO augmentation pipeline to include 
# your specific "dead pixel" requirements.
# add **kwargs to handle the 'transforms' argument from YOLO
def custom_albumentations_init(self, p=1.0, **kwargs):
    """
    Monkey-patch function to inject Salt-and-Pepper noise (Spicy Pixels).
    """
    self.p = p
    self.contains_spatial = False #set to false because blur/noise does not move the bounding boxes
    self.transform = A.Compose([
        A.Blur(p=0.1),
        A.MedianBlur(p=0.1),
        
        # --- THE SPICY PIXEL INJECTION ---
        # --- NEW & IMPROVED: PIXEL DROPOUT ---
        # This randomly turns 1% of pixels into dead (0) pixels.
        # It is cleaner and less error-prone than CoarseDropout.
        A.PixelDropout(
            dropout_prob=0.01,  # 1% of pixels in the image will die
            drop_value=255,       # 0 = Black/Cold dead pixels, 255 for hot pixels, spicy variety
            p=0.5               # Apply this to 50% of images
        ),
        # -------------------------------------
        
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01)
    ], p=p)

# Apply the patch to YOLO's internal class
augment.Albumentations.__init__ = custom_albumentations_init


if __name__ == '__main__':
    # 2. Load the model
    model = YOLO("yolo11s.pt")
    
    # 3. Train with Correct Parameter Placement
    results = model.train(
        data="data.yaml", 
        name='run1',
        
        # Training Logistics
        epochs=600, 
        patience=50, 
        imgsz=640, 
        device=0,   # force script to use gpu
        batch=64,   # increase batch size from 8 to fill the 32GB Tesla gpu memory
        workers=8,  # speed up data loading
        
        # --- THERMAL HYPERPARAMETERS (Must be INSIDE the parentheses) ---
        
        # Sensor Drift (Brightness/Contrast)
        hsv_h=0.0,      # Disable Hue (Physics consistency)
        hsv_s=0.0,      # Disable Saturation
        hsv_v=0.5,      # High Brightness variance (Sensor offset)
        
        # Geometry & Distance
        degrees=10.0,   # Rotation
        translate=0.1,  # Shift
        scale=0.6,      # Zoom variance
        shear=0.0,      # Low shear
        
        # Advanced & Noise
        mosaic=1.0,     # Critical for small objects
        mixup=0.15,     # Helps with faint signals
        copy_paste=0.0, # Disable
        
        # Packet Loss Simulation
        # 'erasing' creates larger black rectangles (occlusion/dropped packets)
        # This works WITH the spicy pixels (which are single dots)
        erasing=0.4     
    )