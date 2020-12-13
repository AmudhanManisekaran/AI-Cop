import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate
# from find import find

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
# showSteps = True

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return

    current_dir = os.path.dirname(os.path.realpath(__file__))
    vehicle_dataset = os.path.join(current_dir, 'detected_vehicles')

    # vehicle_img  = cv2.imread("detected_vehicles/object23.png")
    # for each in range(1,19):
    path, dirs, files = next(os.walk(vehicle_dataset))
    file_count = len(files)

    for each in range(1,file_count):
        vehicle_img_dir = os.path.join(vehicle_dataset, 'object ('   +   str(each) + ').png')
        vehicle_img = cv2.imread(vehicle_img_dir)
        # vehicle_img  = cv2.imread(      'img1 ('    +  str(each_number)   +   ').png'     )

        if vehicle_img is None:
            print("\nerror: image not read from file \n\n")
            os.system("pause")
            return

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(vehicle_img)           # detect plates

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

        # cv2.imshow("vehicle_img", vehicle_img)            # show scene image

        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")
            cv2.imwrite("license_plate/no_plates" + str(each) + ".png", vehicle_img)
            cv2.imwrite("license_plate/result/no_plates" + str(each) + ".png", vehicle_img)
        else:
                    # sort the list of possible plates
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

            licPlate = listOfPossiblePlates[0]

            # cv2.imshow("imgPlate", licPlate.imgPlate)
            print("Image saved")

            cv2.imwrite("license_plate/result/vehicle" + str(each) + ".png", vehicle_img)
            cv2.imwrite("license_plate/vehicle" + str(each) + ".png", vehicle_img)
            cv2.imwrite("license_plate/vehicle" + str(each) + "plate.png", licPlate.imgPlate)
            # cv2.imwrite("vehicle_num_ext/license_plate/vehicle" + str(time) + "plate.png", licPlate.imgPlate)
            cv2.imwrite("license_plate/detected/vehicle" + str(each) + "plate.png", licPlate.imgPlate)

            drawRedRectangleAroundPlate(vehicle_img, licPlate)
            print("----------------------------------------")

            # cv2.imshow("vehicle_img", vehicle_img)
    cv2.waitKey(0)

    os.system('python find.py')
    
    return
###################################################################################################
def drawRedRectangleAroundPlate(vehicle_img, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    cv2.line(vehicle_img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(vehicle_img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(vehicle_img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(vehicle_img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
###################################################################################################
if __name__ == "__main__":
    main()
