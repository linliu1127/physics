import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def savitzky_golay(signal_array, window_length = 3, polyorder = 2): #https://basdas.in/savitzky-golay-filter-in-python-using-numpy-and-scipy/
    assert window_length % 2 == 1, "window length should be odd"
    assert polyorder < window_length, "polynomial order should be less than window length"
    window_indices = np.arange(0, window_length, 1) - np.floor(window_length/2)
    window_indices = window_indices.reshape(-1, 1)
    j_matrix = np.hstack([window_indices**i for i in range(polyorder+1)])
    C=np.linalg.inv(j_matrix.T.dot(j_matrix)).dot(j_matrix.T)
    filter_coefficients = C[0] #(J^TJ)^-1J^T
    filtered_signal = np.convolve(signal_array, filter_coefficients, 'same')
    return(filtered_signal,C)

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100,bias=166):


    B = brightness / 255.0
    c = contrast / 255.0 

    # img = 10*np.tan(img/bias*np.pi/4)
    img = 100*(img/bias)**10
    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('./zeeman effect/zeeman week2_ver2/normal/image2.jpg', cv2.IMREAD_GRAYSCALE)
row , col=img.shape
dimg=np.zeros((1,col))
window=11
order=2
cut=img[int(row/2),:]
p0=plt.plot(np.arange(col),cut[0:],5,label='origin')
dcut=np.zeros(np.shape(cut))
ddcut=np.zeros(np.shape(cut))
cut,C=savitzky_golay(cut,window,order)
for i in range(int((window+1)/2),col-int((window+1)/2)):
    vec=cut[i-int((window+1)/2):i+int((window+1)/2)-1]
    a=np.dot(C,vec) #a is coefficient of local polynomial
    cut[i]=a[0]
    dcut[i]=a[1]
    ddcut[i]=2*a[2]
# modify_contrast_and_brightness2(img)
nddcut=np.where(ddcut<0,True,False)
zerodcut=np.where((dcut<1) & (dcut>-1),True,False)
target=nddcut & zerodcut                        #for np array & is the and operation of array
p1=plt.plot(np.arange(col),cut[0:],5,label='cut')
p2=plt.plot(np.arange(col),dcut[0:],label='dcut')
p3=plt.plot(np.arange(col),ddcut[0:],label='ddcut')
p4=plt.scatter(np.where(target)[0],cut[target],10,label='target')
plt.plot([0,col],[0,0])
plt.legend(loc="upper left")

plt.show()
# for i in range(1,col-4):
#     dimg[0][i+2]=(img[int(row/2)][i+4]+8*img[int(row/2)][i+3]-8*img[int(row/2)][i+1]+img[int(row/2)][i])/12
# plt.plot(np.arange(col),dimg[0][0:])
# plt.show()