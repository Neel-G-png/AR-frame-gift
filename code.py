from flask import Flask, render_template, Response, request
from cv2 import cv2
import datetime, time
import os, sys
import numpy as np


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass

#Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__)

# im_src = cv2.imread("new_scenery.jpg")
camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    camera = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture("videoplayback.mp4")
    counter = 0
    while True:
        success, frame = camera.read()
        ret1,im_src = cap2.read() 
        if not ret1:
            exit()

        if success:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            parameters =  cv2.aruco.DetectorParameters_create()
            try:
                markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
                
                if len(markerIds) == 4:
                    counter = 0
                    # print(len(markerIds),end = "\r")
                    index = np.squeeze(np.where(markerIds==25))
                    refPt1 = np.squeeze(markerCorners[index[0]])[1]
                    index = np.squeeze(np.where(markerIds==33))
                    refPt2 = np.squeeze(markerCorners[index[0]])[2]

                    distance = np.linalg.norm(refPt1-refPt2)
                    
                    scalingFac = 0.02
                    pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
                    pts_dst = pts_dst + [[refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]]
                    
                    index = np.squeeze(np.where(markerIds==30))
                    refPt3 = np.squeeze(markerCorners[index[0]])[0]
                    pts_dst = pts_dst + [[refPt3[0] + round(scalingFac*distance), refPt3[1] + round(scalingFac*distance)]]

                    index = np.squeeze(np.where(markerIds==23))
                    refPt4 = np.squeeze(markerCorners[index[0]])[0]
                    pts_dst = pts_dst + [[refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]]

                    pts_src = [[0,0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]
                    
                    pts_src_m = np.asarray(pts_src)
                    pts_dst_m = np.asarray(pts_dst)

                    h, status = cv2.findHomography(pts_src_m, pts_dst_m)
                    
                    # Warp source image to destination based on homography
                    warped_image = cv2.warpPerspective(im_src, h, (frame.shape[1],frame.shape[0]))

                    mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

                    # Erode the mask to not copy the boundary effects from the warping
                    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                    mask = cv2.erode(mask, element, iterations=3)
                    warped_image = warped_image.astype(float)
                    mask3 = np.zeros_like(warped_image)
                    for i in range(0, 3):
                        mask3[:,:,i] = mask/255

                    # Copy the warped image into the original frame in the mask region.
                    warped_image_masked = cv2.multiply(warped_image, mask3)
                    frame_masked = cv2.multiply(frame.astype(float), 1-mask3)
                    im_out = cv2.add(warped_image_masked, frame_masked)
                    
                    # Showing the original image and the new output image side by side
                    concatenatedOutput = cv2.hconcat([frame.astype(float), im_out])   
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(concatenatedOutput,1))
                    concatenatedOutput = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + concatenatedOutput + b'\r\n') 
                elif len(markerIds)<4:
                    counter = 0
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                # print(e)
                time.sleep(1)
                counter+=1
                if counter>=10:
                    camera.release()
                    cap2.release()
                    # p.stop()
                    # print("TIME OUT !!!!!!!!!!")
                    break
                
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)