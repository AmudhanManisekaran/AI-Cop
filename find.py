import os
import numpy as np
import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from sklearn.externals import joblib

############################ prediction ########################################

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
plate_dir = os.path.join(current_dir, 'license_plate/detected/')
total_dir = os.path.join(current_dir, 'license_plate/result/')

model = joblib.load(model_dir)



############################ prediction end ########################################

############################ localization ######################################

#####################################function definition##########################################

def extract_text(car_image):

    print(car_image.shape)

    gray_car_image = car_image * 255
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gray_car_image, cmap="gray")
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    ax2.imshow(binary_car_image, cmap="gray")
    plt.show()

    ############################ localization end######################################

    ############################ cca  ######################################
    label_image = measure.label(binary_car_image)
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray");

    plate_like_objects = []
    plate_objects_cordinates = []

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        #if region.area < 3400 or region.area > 10000:
        # if region.area < 19000 or region.area > 23000:
        if region.area < 2000:
            #if the region is so small then it's likely not a license plate
            continue

        # the bounding box coordinates
        minRow, minCol, maxRow, maxCol = region.bbox

        plate_like_objects.append(binary_car_image[minRow:maxRow,
                                  minCol:maxCol])
        plate_objects_cordinates.append((minRow, minCol,
                                              maxRow, maxCol))

        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        # let's draw a red rectangle over those regions

    plt.show()

    ############################ cca  end ##########################################

    ############################ segmentation ##########################################


    # print("plate_like_objects",plate_like_objects)


    if(plate_like_objects):
        pass
    else:

        print("hello")
        return None

    license_plate = np.invert(plate_like_objects[0])

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")

    character_dimensions = (0.35*license_plate.shape[0], 0.9*license_plate.shape[0], 0.01*license_plate.shape[1], 0.9*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        # print("region_height",region_height)
        region_width = x1 - x0
        # print("region_width",region_width)

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]


            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                           linewidth=2, fill=False)
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)

    plt.show()

    ############################ segmentation end ##########################################
    ############################ prediction ########################################
    classification_result = []
    for each_character in characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = model.predict(each_character)
        classification_result.append(result)

    # print(classification_result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    # print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    print(rightplate_string)

    time=datetime.datetime.now()
    # print(time)
    # print(car_image)

    file = open('plates.txt','a')
    file.write(    str(time) +"      " + rightplate_string +"\n" )
    file.close()

    # os.rename(car_image,rightplate_string)
    ############################ prediction ########################################



# car_dir = os.path.join(current_dir,'license_plate/result')

path, dirs, files = next(os.walk(total_dir))
file_count = len(files)
print("file_count",file_count)

file = open('plates.txt','a')
file.write( "\n" )
file.close()


for each_number in range(1,file_count+1):
# for each_number in range(1,17):

    try:
        car_image = imread(   plate_dir+ "vehicle"    +  str(each_number)   +   'plate.png', as_grey=True     )
        p=extract_text(car_image)
        print(str(each_number))
    except:
        pass

    # print("car_image",car_image)
    # if(car_image):
    #     pass
    # else:
    #     break

    # car_image = imread("input_img\plate1.png", as_grey=True)

    # p=extract_text(car_image)
