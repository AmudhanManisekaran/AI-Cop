import cv2
import os

vehicle_count = [0]

current_path = os.getcwd()

def save_image(source_image):
	# cv2.imwrite(current_path + "/detected_vehicles/vehicle" + str(len(vehicle_count)) + ".png", source_image)
	cv2.imwrite(current_path + "/detected_vehicles/object (" + str(len(vehicle_count)) + ").png", source_image)
	# vehicle_img_dir = os.path.join(vehicle_dataset, 'object ('   +   str(each) + ').png')
	vehicle_count.insert(0,1)
