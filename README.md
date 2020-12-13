# AI-Cop
An artificially intelligent traffic violation detection and management system.

## What does it do?
- Identifies vehicles jumping traffic lights and capture their images.
- Locates the license plate in the captured image
- Generates a Machine learning model for Optical Character Recognition
- Uses the generated model to extract text from the identified license plates
- Stores the details of the violator along with the timestamp for further action

## How it does that?

## Phase 1: Identify violations

![identify](/readme_img/Picture2.png)

## Phase 2: Detect the violating vehicle

![identify1](/readme_img/Picture3.png)

![identify2](/readme_img/Picture4.png)

## Phase 3: Image processing to identify license plate

![imgpro1](/readme_img/Picture5.PNG)

![imgpro2](/readme_img/Picture6.PNG)

![imgpro3](/readme_img/Picture7.PNG)

![imgpro4](/readme_img/Picture8.PNG)

## Phase 4: Optical character recognition

![ocr1](/readme_img/Picture9.png)

## Phase 5: Output

```bash
    Date        Time                License no.
    2019-02-11  10:13:35.276620     TN66QS113
```

## Requirements:
- Python
- Tensorflow
- cv2
- numpy
- pytesseract
- imutils
- utils
- csv
- math

## How to run it?
1. Place the traffic surveillance video in the 'input_footage' folder inside src
2. Update 'input_video' variable in 'Vehicle_detection.py'. You can also use a live footage from an installed surveillance camera.
3. Detect the violators by executing
```bash
	python Vehicle_detection.py
```
4. The pictures of the violating vehicles will be placed in the 'detected_vehicles' folder.
5. Obtain the license numbers of those vehicles by executing
```bash
	python Main.py
```
6. All the license numbers, along with the timestamp are stored in 'plates.txt'. The pictures of all the license plates will be found in the 'license_plate' folder.
