#build by MUS(mohammad usman sharif)
#and he's team
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import face_recognition
import mysql.connector
import smtplib
from email.message import EmailMessage
from twilio.rest import Client

#function to send emails
def email_alert(subject, body, to):
    mesg = EmailMessage()
    mesg.set_content(body)
    mesg['subject'] = subject
    mesg['to'] = to
    #email that used to send mails
    user = "XXXXXXXXXX@gmail.com"
    mesg['from'] = "Real Time Mask"
    #pass is app password of the email
    password = "ASpadadhbclicnc"
    #port no 587
    server = smtplib.SMTP("smtp.gmail.com", 587)
    
    server.starttls()
    server.login(user, password)
    server.send_message(mesg)
    server.quit()


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (lcs, preds)

#connection establishment
mydb =mysql.connector.connect(host="localhost", user="root", passwd="123456", database="personaldb")

if mydb.is_connected() == True:
    print("[INFO] DataBase Connection is Successfully")
#creating a cursor
mycur = mydb.cursor()

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

#loading our for face recognition images
path = 'image'

images = []
#taking name from image file
names = []

mylist = os.listdir(path)
# it will print list of images
#print(mylist)

#importing images with name
for r in mylist:
	curImg = cv2.imread(f'{path}/{r}')
	images.append(curImg)
	names.append(os.path.splitext(r)[0])
print("[INFO] Names Loading Complete")

def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList

encodeListKnown = findEncodings(images)
print("[INFO] Encoding Complete")

# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

#email body
sub = "Not Wearing Mask Alert"
bod = "As our system has detected that you have not weared a mask Since COVID-19 has spread worldwide and played havoc with the lives of people, we, therefore, had devised the mask-wearing strategy to deal with the danger in an effective way but your non-compliance with the instructions is an indicator of the fact that you are not taking the instructions seriously and putting others in danger along with yourself which is not acceptable at all here. so wear a mask and Stay safe"

aid = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=900)


	imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
	imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

	faceCurLoc = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS,faceCurLoc)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		if mask > withoutMask:
			# determine the class label and color we'll use to draw
			label = "MASK"
			color = (0, 255, 0)
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		else:
			# determine the class label and color we'll use to draw
			label = "No MASK"
			color = (0, 0, 255)
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			# display the label and bounding box rectangle on the output
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			
			for encodeface,faceLoc in zip(encodesCurFrame,faceCurLoc):
				#Recognizing face using face_recodnition library
				matches = face_recognition.compare_faces(encodeListKnown,encodeface)
				faceDis = face_recognition.face_distance(encodeListKnown,encodeface)
				#print(faceDis)
				matchIndex = np.argmin(faceDis)
				#finding the name of matched face				
				if matches[matchIndex]:
					name = names[matchIndex].upper()
					print(name)
					cv2.rectangle(frame,(startX,endY-35),(endX,endY),(0,0,255),cv2.FILLED)
					cv2.putText(frame,name,(startX+6,endY-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
					p="select email, Ph_no, id from Users where name = %s"
					mycur.execute(p,(name, ))
					row = mycur.fetchall()
					de = row[0][0]
					be = row[0][1]
					qid = row[0][2]
					#print(be)
					pto = "+91" + be
					if aid == 0:
						print(de)
						email_alert(sub, bod, de)
						print("[INFO] Email Send Successfully.....")
						print(pto)
						#enter your twillio account sid and token
						account_sid = "AC045rwshsfsdnd52f92058b19bd18cbb"
						auth_token = "b95c16sasksjdcy09dcbcj2cf4c0668d481"
						client = Client(account_sid, auth_token)

						message = client.messages \
										.create(
											body="As our system has detected that you have not weared a mask Since COVID-19 has spread worldwide and played havoc with the lives of people, we, therefore, had devised the mask-wearing strategy to deal with the danger in an effective way but your non-compliance with the instructions is an indicator of the fact that you are not taking the instructions seriously and putting others in danger along with yourself which is not acceptable at all here. so wear a mask and Stay safe",
											#number that you get after creating twillio account
											from_='+1252678124821',
											to=pto
										)

						print(message.sid)
						print("[INFO] SMS Send Successfully.....")
						aid = qid
					elif qid != aid:
						print(de)
						email_alert(sub, bod, de)
						print("[INFO] Email Send Successfully.....")
						print(pto)
						account_sid = "AC045rwshsfsdnd52f92058b19bd18cbb"
						auth_token = "b95c16sasksjdcy09dcbcj2cf4c0668d481"
						client = Client(account_sid, auth_token)

						message = client.messages \
										.create(
											body="As our system has detected that you have not weared a mask Since COVID-19 has spread worldwide and played havoc with the lives of people, we, therefore, had devised the mask-wearing strategy to deal with the danger in an effective way but your non-compliance with the instructions is an indicator of the fact that you are not taking the instructions seriously and putting others in danger along with yourself which is not acceptable at all here. so wear a mask and Stay safe",
											from_='+1252678124821',
											to=pto
										)

						print(message.sid)
						print("[INFO] SMS Send Successfully.....")
						aid = qid


	# show the output frame
	cv2.imshow("VideoStream", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()