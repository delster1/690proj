# EECS 690 Project - Animal Spotter 

## Idea:
Create a model that analyzws photos to detect and identify animals 

Example usage:
1. Point camera at a animal 
2. Camera auto-captures pictures every so often
3. Model automatically processes photo
4. Outputs classification

Materials:
- Jetson Nano
- Sony ZVE1 Camera

### Details
Model is always running inference on most recent photo taken
- Queue up pictures as we get them
- Once model finished processing one picture, output result and process next

On picture taken:
1. Start inference on new picture
2. Output classification
3. If new picture present, start inference, else wait until picture taken

#### Training Data:
Only using a handful of animals initially

#### Model:
CNN 

## Tentative Plan / List of things todo

1. Find Training Data source
2. Create training data
3. Create Basic Model
4. Create basic model / camera controller - run and process every so often
5. Figure out output
6. Refine Model
7. Create Presentation
