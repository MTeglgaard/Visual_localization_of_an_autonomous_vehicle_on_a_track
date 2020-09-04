#####################################################################################
#
#This code has been developed for a master project at DTU, by Michael Teglgaard, 2020
#
#The code was build on code from a tutorial from STEREOLABS

########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

#https://github.com/stereolabs/zed-examples/tree/master/svo%20recording/playback

"""
    Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as
    a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.
"""
import sys
import pyzed.sl as sl
import cv2
import math
import time
from laneFunc import *

def main():

	recordVideo = bool(False)
	showImages = bool(False)

	if len(sys.argv) != 2:
		print("Please specify path to .svo file.")
		exit()

	filepath = sys.argv[1]
	print("Reading SVO file: {0}".format(filepath))


	# Specify SVO path parameter
    # init_params = sl.InitParameters()
    # init_params.set_from_svo_file(str(svo_input_path))
    # init_params.svo_real_time_mode = False  # Don't convert in realtime
    # init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)


	input_type = sl.InputType()
	input_type.set_from_svo_file(filepath)
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.coordinate_units = sl.UNIT.CENTIMETER  # Use milliliter units (for depth measurements)
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
	init_params.depth_minimum_distance = 1 # Set the minimum depth perception distance to 1 m
	init_params.depth_maximum_distance = 40 # Set the maximum depth perception distance to 40m


	cam = sl.Camera()
	
	status = cam.open(init_params)
	if status != sl.ERROR_CODE.SUCCESS:
		print(repr(status))
		exit()

	runtime = sl.RuntimeParameters()
	image = sl.Mat()
	depth_map = sl.Mat()
	point_cloud = sl.Mat()

	## initialize image window
	#define the screen resulation
	screen_res = 1920,1200
	#image size
	width = cam.get_camera_information().camera_resolution.width
	height = cam.get_camera_information().camera_resolution.height 
	


	scale_width = screen_res[0] / width
	scale_height = screen_res[1] / height
	scale = min(scale_width, scale_height)
	#resized window width and height
	window_width = int(width * scale)
	window_height = int(height * scale)
	window_size = (window_width, window_height)
	
	#cv2.WINDOW_NORMAL makes the output window resizealbe
	cv2.namedWindow('result', cv2.WINDOW_NORMAL)

	#resize the window according to the screen resolution
	#cv2.resizeWindow("result", window_width, window_height)


	key = ''
	print("  Save the current image:     s")
	print("  Quit the video reading:     q\n")


	print(cam.get_svo_position())
	cam.set_svo_position(10000)
	print(cam.get_svo_position())
	i=0
	frame_processing_time = time.time()

	while (key != 113) or (i!=50):  # for 'q' key
		err = cam.grab(runtime)
		if err == sl.ERROR_CODE.SUCCESS:
			cam.retrieve_image(image, sl.VIEW.LEFT)
			# To recover data from sl.Mat to use it with opencv, use the get_data() method
			# It returns a numpy array that can be used as a matrix with opencv
			frame = image.get_data()
			# Get the depth map
			cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

			cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
			#line detection
			lane_image = np.copy(frame)
			canny_image = canny(frame)
			cropped_image = region_of_intrest(canny_image)
			lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]), minLineLength=40, maxLineGap=5)
			averaged_lines = average_slope_intercept(lane_image, lines)
			Hline_image = display_lines(lane_image, lines)
			line_image = display_lines(lane_image, averaged_lines)
			combo_image = cv2.addWeighted(frame,0.8, line_image, 1, 1)
			cropped_combo_image = region_of_intrest(combo_image)


			## get distance / location



            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
			x = round(image.get_width() / 2)
			y = round(image.get_height() / 2)
			err, point_cloud_value = point_cloud.get_value(x, y)
			distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
								point_cloud_value[1] * point_cloud_value[1] +
								point_cloud_value[2] * point_cloud_value[2])

			point_cloud_np = point_cloud.get_data()
			# point_cloud_np.dot(tr_np)

			if not np.isnan(distance) and not np.isinf(distance):
				# print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
                # Increment the loop
				i = i + 1
			else:
				print("Can't estimate distance at this position.")                
				print("Your camera is probably too close to the scene, please move it backwards.\n")
			sys.stdout.flush()


			# Get and print distance value in mm at the center of the image
			# We measure the distance camera - object using Euclidean distance
			# x = mat.get_width() / 2
			# y = mat.get_height() / 2
			# point_cloud_value = point_cloud.get_value(x, y)

			# distance = math.sqrt(point_cloud_value[0]*point_cloud_value[0] + point_cloud_value[1]*point_cloud_value[1] + point_cloud_value[2]*point_cloud_value[2])
			# printf("Distance to Camera at (", x, y, "): ", distance, "mm")

			# x1, y1, x2, y2 = lines[0].reshape(4)
			# depth_value = depth_map.get_value(x2,y2)
			# print("x2= ",x2," y2= ",y2," z= ",depth_value)
			# point3D = point_cloud.get_value(1220,850)#1220	850
			# x = point3D[0]
			# y = point3D[1]
			# z = point3D[2]
			# #color = point3D[3]
			# print("x= ",x," y= ",y," z= ",z)
			
			if showImages:
				# cv2.imshow("cropped_lane_image", cropped_lane_image)
				# frame_canny = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)
				# frame_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

				# plt.imshow(cropped_combo_image)#canny_image)
				# plt.show()

				# combined_image= combine(lane_image, cropped_combo_image ,canny_image, cropped_image, combo_image  , Hline_image )# , line_image
				#cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
				#com_height, com_width, com_layers = combined_image.shape
				#imS = cv2.resize(combined_image, window_size)
				#cv2.imshow("combined_image",imS)
				#cv2.imshow("result", combo_image)
				# cv2.imshow("result", combined_image)
				cv2.imshow("result", cropped_combo_image)
				# end line detection

				#cv2.imshow("ZED", mat.get_data())
				key = cv2.waitKey(1)
				saving_image(key, mat)
				# print(int(cam.get_svo_number_of_frames()//2))
				# cam.set_svo_position(10000)
			else:
				key=113
		else:
			key = cv2.waitKey(1)
		# i= i+1

		#measure the time it takes to process one frame
		print("--- %s seconds ---" % (time.time() - frame_processing_time))
		frame_processing_time = time.time()

	cv2.destroyAllWindows()

	print_camera_information(cam)
	cam.close()
	print("\nFINISH")


def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            print()
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_runtime_parameters().confidence_threshold))
            print("Depth min and max range values: {0}, {1}".format(cam.get_init_parameters().depth_minimum_distance,
                                                                    cam.get_init_parameters().depth_maximum_distance)
)
            print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
            print("Camera FPS: {0}".format(cam.get_camera_information().camera_fps))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def saving_image(key, mat):
    if key == 115:
        img = sl.ERROR_CODE.FAILURE
        while img != sl.ERROR_CODE.SUCCESS:
            filepath = input("Enter filepath name: ")
            img = mat.write(filepath)
            print("Saving image : {0}".format(repr(img)))
            if img == sl.ERROR_CODE.SUCCESS:
                break
            else:
                print("Help: you must enter the filepath + filename + PNG extension.")


def getworldCordinates(x, y, point_cloud):
	point_cloud_value = point_cloud.get_value(x, y)
	distance = math.sqrt(point_cloud_value[0]*point_cloud_value[0] + point_cloud_value[1]*point_cloud_value[1] + point_cloud_value[2]*point_cloud_value[2])

if __name__ == "__main__":
    main()
