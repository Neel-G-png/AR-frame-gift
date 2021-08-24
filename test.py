import cv2 as cv2
import numpy as np
import time
# from ffpyplayer.player import MediaPlayer
import vlc

cap = cv2.VideoCapture(0)
im_src = cv2.imread("new_scenery.jpg")
cap2 = cv2.VideoCapture("videoplayback.mp4")
p = vlc.MediaPlayer("cp.mp3")
# audio = MediaPlayer("videoplayback.mp4")
while True:
    ret,frame = cap.read()
    ret1,im_src = cap2.read()
    # audio_frame,val = audio.get_frame()
    frame = cv2.resize(frame,(640,480))
    # cv2.imshow("test",im_src)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    try:
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(len(markerIds))
        # p.play()
        if markerIds:
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
            cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
            # if val != 'eof' and audio_frame is not None:
        # #audio
        #     img, t = audio_frame
        # else:
        #     time.sleep(5)
        #     cap.release()
        #     cap2.release()
            # p.stop()
            # cv2.imshow("AR using Aruco markers", frame)
        
    except:
        # print("waitng")
        pass
    
    cv2.imshow("AR using Aruco markers",frame)
    if cv2.waitKey(13) == 1:
        break

cv2.destroyAllWindows()

# import cv2
# import numpy as np
# #ffpyplayer for playing audio
# from ffpyplayer.player import MediaPlayer
# video_path="test_vid.webm"
# def PlayVideo(video_path):
#     video=cv2.VideoCapture(video_path)
#     player = MediaPlayer(video_path)
#     while True:
#         grabbed, frame=video.read()
#         audio_frame, val = player.get_frame()
#         if not grabbed:
#             print("End of video")
#             break
#         if cv2.waitKey(28) & 0xFF == ord("q"):
#             break
#         cv2.imshow("Video", frame)
#         if val != 'eof' and audio_frame is not  :
#             #audio
#             img, t = audio_frame
#     video.release()
#     cv2.destroyAllWindows()
# PlayVideo(video_path)