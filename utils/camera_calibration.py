import numpy as np
import cv2 as cv
import glob
import argparse
import platform
import os
import json
from camera import camera


def main():
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--output_dir', type=str, default='data/camera', help='Output directory')
    parser.add_argument('--pattern', type=str, default='8x6', help='Pattern size')
    parser.add_argument('--square_size', type=int, default=1, help='Square size')

    args = parser.parse_args()

    output_dir = args.output_dir
    pattern = args.pattern.split('x')
    pattern = (int(pattern[0]), int(pattern[1]))
    square_size = args.square_size

    cam = camera()
    machine_name = platform.mac_ver()
    machine_name = machine_name[0] + machine_name[2]

    output_file = os.path.join(output_dir, machine_name + '.json')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    recording = True

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    count = 0
    while recording:
        key = input("1 - Take photo\n2 - Stop\nSelect: ")
        if key == '1':
            ok, photo = cam.get_frame()
            if not ok:
                print('No frame')
            else:
                gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, pattern, None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    print(f'Chessboard n_{count+1} found')
                    count += 1
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    cv.drawChessboardCorners(photo, pattern, corners2, ret)
                    cv.imshow('chessboard pattern', photo)
                    cv.waitKey(500)

                    if count >= 10:
                        print('10 photos taken, ready to calibrate')

                else:
                    print('No chessboard found')

            input('\nPress any key to continue')
            os.system('cls' if os.name == 'nt' else 'clear')
        elif key == '2':
            recording = False
            os.system('cls' if os.name == 'nt' else 'clear')

        else:
            print('Invalid input')
            os.system('cls' if os.name == 'nt' else 'clear')

    if count >= 10:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        print('Calibration finished: ', ret)
        print('Camera matrix:\n', mtx.tolist())
        print('Distortion coefficients:\n', dist.tolist())
        print('Rotation vectors:\n', rvecs[0].tolist())
        print('Translation vectors:\n', tvecs[0].tolist())

        data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'rotation_vectors': rvecs[0].tolist(),
            'translation_vectors': tvecs[0].tolist()
        }

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("\ntotal error: {}".format(mean_error/len(objpoints)))

        key = input("Save calibration? (y/n): ")

        if key == 'y' or key == '' or key == 'Y':
            json_string = json.dumps(data)
            with open(output_file, 'w') as outfile:
                outfile.write(json_string)

    else:
        print('Not enough photos taken, stopping')


if __name__ == "__main__":
    main()
