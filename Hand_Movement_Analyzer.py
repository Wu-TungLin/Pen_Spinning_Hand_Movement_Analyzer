import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


wrist = []
index_finger_mcp = []

def draw(data):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(data)[:,0], np.array(data)[:,1], s=20, alpha=0.4, linewidths=2.5, c='#AAAFFF', edgecolors='blue')
    ax.invert_yaxis()
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.show()

def drawsd(data, handsize, sd):
    x = []
    y = []
    final = []
    
    for i in range(len(data)-15+1):
        y.append(SD(data[i:i+14], handsize))
        x.append(i+7.5)
        final.append(sd)
        
    plt.plot(x, y, label = "Movement per 15 frames")
    plt.plot(x, final, "r", label = "Final result" )
    plt.xlabel("Frame")
    plt.ylabel("Standard deviation (Normalized)")
    plt.legend()
    plt.show()

def drawl2(data, handsize, L2):
    x = []
    final = []

    for i in range(len(data)):
        x.append(i)
        final.append(L2)
        
    plt.plot(x, np.array(distance(data))/handsize, label = "Movement per frame")
    plt.plot(x, final, "r", label = "Final result")
    plt.xlabel("Frame")
    plt.ylabel("Distance (Normalized)")
    plt.legend()
    plt.show()

def avgdistance(data1, data2):
    
    result = 0
    for i in range(len(data1)):
        result += ((data1[i][0] - data2[i][0])**2 + (data1[i][1] - data2[i][1])**2)**0.5
    
    result /= len(data1)
    return result


def distance(data):
    cx, cy = np.mean(data, axis=0)   
    result = []
    for i in range(len(data)):
        
        result.append(((cx - data[i][0])**2 + (cy - data[i][1])**2)**0.5)
        
    return result
    
      
def L2norm(data1, data2):
    
    result = np.mean(distance(data1), axis = 0)
    result /= avgdistance(data1,data2)
    
    return result
 
    
def SD(data, handsize):
    cx, cy = np.mean(data, axis = 0)
    sigmax = 0
    sigmay = 0
    
    for i in range(len(data)):
        sigmax += (data[i][0] - cx)**2
        sigmay += (data[i][1] - cy)**2
        
    sigmax = (sigmax/len(data))**0.5/handsize
    sigmay = (sigmay/len(data))**0.5/handsize
    sigma = (sigmax**2+sigmay**2)**0.5
    
    return sigma


def main(filename, confidence):
    finish = "Break"
    cap = cv2.VideoCapture(filename)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence = confidence)   
    mpDraw = mp.solutions.drawing_utils
       
    
    while True:
        success, img = cap.read()
        
        if not success:
            break
        finish = "True"
        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        if imgHLS[:,:,(1)].mean() > 120:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
            s_magnification = 0.5
            v_magnification = 1
            imgHSV[:,:,(1)] = imgHSV[:,:,(1)]*s_magnification  
            imgHSV[:,:,(2)] = imgHSV[:,:,(2)]*v_magnification
            imgRGB = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
            
        else: 
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        results = hands.process(imgRGB)
        
        
        if results.multi_hand_landmarks:
    
    
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
    
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id == 0: 
                        wrist.append([cx, cy])
                    if id == 5:
                        index_finger_mcp.append([cx, cy])
    
        
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
                   
        if key == 27:
            finish = "KB"
            break
        
    cv2.destroyAllWindows()
    cap.release()
    
    return finish


print("This Hand movement analyzing program is made by: ")
print("Dove from Taiwanese Pen Spinning Forum (TWPS)")
print("Wabi from JapEn Board (JEB)")
print()
while True:
    wrist = []
    index_finger_mcp = []
    

    print('Please enter "q" if you want to close this program')
    print()
    print("Please enter the video path, remember to type the filename extension")
    print("For example:", r"C:\desktop\Penspinning\xxx.mp4")
    print()
    print('If you want to close the window of the analyzing video to stop calculating, please press "esc"')
    
    
    filepath = input()
    if filepath == "q" or filepath == "Q":
        break
    
    print()
    print("----------------")
    print("Do you want to change the sensibility of the model?")
    print("The default number of the sensibility is 0.25")
    print("The higher the number, the higher the sensibility")
    print('Enter "y" if yes, enter "n" if no')
    flag = False
    while not flag:
        choice = input()
        if choice == "y" or choice == "Y":
            flag = True
            print()
            print("----------------")
            print("The sensibility is a number between 0~1")
            sens = -1
            while sens<0 or sens>1:
                try:
                    sens = 1-float(input())
                    if sens<0 or sens>1:
                        print("Please enter a number between 0~1")
                except:
                    print("Please enter a number between 0~1")

        elif choice == "n" or choice == "N":
            flag = True
            sens = 0.75
        else:
            print('Enter "y" if yes, enter "n" if no')

    try:
        finish = main(filepath, sens)
        if finish == "True":
            if len(wrist) == 0:
                print()
                print("The model cannot detect anything, please higher the sensibility")
            else:
                draw(wrist)
                L2 = L2norm(wrist, index_finger_mcp)
                sd = SD(wrist, avgdistance(wrist, index_finger_mcp))                
                print("----------------")
                print("L2 norm:", L2)
                drawl2(wrist, avgdistance(wrist, index_finger_mcp), L2)
                print("Standard deviation:", sd)
                drawsd(wrist, avgdistance(wrist, index_finger_mcp), sd)
        elif finish == "KB":
            print()
            print("Keyboard interrupt")
        elif finish == "Break":
            print()
            print("Please enter the correct video path")


        print()    
        wrist.clear()
        index_finger_mcp.clear()

    except:
        print("Please enter the correct video path")
    print("----------------")
