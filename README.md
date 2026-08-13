         
**Training Summary**  

data prep for training yolo11s IR detection model on IR imagery with hot pixels ("spicy")

*Folder Structure* 
```text
C:\seals_project\          <-- Main Project Folder
├── train.py               <-- Your script
├── validate.py            <-- Your validation script
├── data.yaml              <-- Your config
│
├── images\                <-- Training imagery folder 
│   ├── train_image_01.png
│   └── ...
│
├── labels\                <-- Labels for training imagery 
    ├── train_image_01.txt  
    └── ...
├── validation\            <-- Validation dataset folder
    ├── images\                 
│       ├── val_image_01.png
│       └── ...
│
    ├── labels\                
        ├── val_image_01.txt  
        └── ...
``` 

Trained on IR imagery from 6 FLIR A6751sc SLS LWIR cameras (640x512) fitted with a 25mm lens collected from an altitude of 800-1200ft. Thermal signatures of animals on arctic sea ice were labeled. Training dataset consists mostly of seals but also includes polar bears, caribou, and foxes.  

Trained yolo11s for speed and accuracy
600 epochs, stop after 50 decline
included speckling ("spice") augmentation setting random pixels to max value (255) 

best.pt weights used in final pipeline

**Final pipeline: 2025_batch_norm_yolo11s.py** 

1) reference manifest of imagelists to process (pathways to IR image lists excluding NUC events and IR frames without color imagery)
2) normalize imagery (despice [replace hot pixel with ave of surrounding values], 0.001% min/max, linear stretch)
3) run spicy v11s best.pt model on image
4) do not save normalized image
5) save detections to VIAME csv format in same flight camera folder with image list

**Model run guidance**  

Run model on raw 16 bit IR imagery. 
Default IR normalization settings that can be adjusted include the following:

DIFF_THRESHOLD = 500  
LOWER_PERCENTILE = 0.001  
UPPER_PERCENTILE = 99.999  

DIFF_THRESHOLD identifies the value change between two pixels that would initiate "despeckling" by replacing this pixel with the average of the surrounding pixel values.  
LOWER_PERCENTILE sets the lower boundary of pixel values for each frame that will be truncated and set as the high boundary for the normalization stretch.  
UPPER_PERCENTILE sets the upper boundary of pixel values for each frame that will be truncated and set as the high boundary for the normalization stretch.  

