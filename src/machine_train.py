import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

# letters = [
#             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
#             'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
#             'U', 'V', 'W', 'X', 'Y', 'Z'
#         ]

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

# numbers = [ 0, 1, 2, 5 ,6, 10, 17, 18 ,25, 26, 29, 30, 33, 37, 38 ,45, 49, 50, 57, 58, 65, 66, 68, 81, 153, 154,
#             155, 156, 157, 158, 161, 166, 168, 169, 170, 173, 174, 175, 176, 177, 178, 181, 182, 185, 186, 187, 189,
#             472, 489, 490, 491, 492, 546, 726, 737, 738, 773, 774, 946, 948, 954, 962, 966, 974, 977, 982, 986, 1010]

# letters = [
#             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
#             'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
#             'U', 'V', 'W', 'X', 'Y', 'Z'
#         ]

# folders = [ "Sample001","Sample002","Sample003","Sample004","Sample005","Sample006",
#             "Sample007","Sample008","Sample009","Sample010","Sample011","Sample012",
#             "Sample013","Sample014","Sample015","Sample016","Sample017","Sample018",
#             "Sample019","Sample020","Sample021","Sample022","Sample023","Sample024",
#             "Sample025","Sample26","Sample27","Sample028","Sample029","Sample030",
#             "Sample031","Sample032","Sample033","Sample034","Sample035","Sample036",]

# numbers = [ "001","002","003","004","005","006",
#             "007","008","009","010","011","012",
#             "013","014","015","016","017","018",
#             "019","020","021","022","023","024",
#             "025","026","027","028","029","030",
#             "031","032","033","034","035","036",]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        # for each_number in numbers:
        for each in range(1,78):
            # image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            image_path = os.path.join(training_directory, each_letter , ' ('   +   str(each) + ').jpg')
            # image_path = os.path.join(training_directory, 'set' + each_number, 'img' + each_number + ' (' + str(each) + ').jpg')



            # read each image of each character
            img_details = imread(image_path, as_grey=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # binary_image = img_details < img_details
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
            # target_data.append(each_number)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


current_dir = os.path.dirname(os.path.realpath(__file__))

# training_dataset_dir = os.path.join(current_dir, 'train')
training_dataset_dir = os.path.join(current_dir, 'train_new_78')
# training_dataset_dir = os.path.join(current_dir, 'fonts')

image_data, target_data = read_training_data(training_dataset_dir)

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='linear', probability=True)

cross_validation(svc_model, 4, image_data, target_data)

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'/svc.pkl')
