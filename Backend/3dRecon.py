import os,sys
import cv2
import numpy as np
import featureExtract as fE

K = np.array([[1.19944599e+03,0.00000000e+00,6.03668941e+02],
            [0.00000000e+00,1.19673729e+03,5.34182170e+02],
            [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
distort = np.array([[-0.06274979, 0.17771384, -0.00146246, 0.00585943, -0.26509556]])
K_inv = np.linalg.inv(K)
# plane-depth -- cm
plane-depth = 1
def computeDesc(gray, pts, size=20):
    '''
        img: gray scale img
        pts: N by 2 np-array of pixel coordinate
        return: desc and kpts array
    '''
    if (len(gray.shape) == 3):
        gray= cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # pre-setup the keypoints and enforce the descriptor computation 
    kps = []
    N = pts.shape[0]
    for i in range(0,N):
        kps.append(cv2.KeyPoint(pts[i,0],pts[i,1],size))
    kpts, des = sift.compute(gray, kps)     
    return kpts,des

def lstsql3d(R, T, x1,x2):
    '''
        T, x1, x2 are K-transfered/homo-vectors, 3 by 1
        T is up to a scale of gamma!
        return x -- 3d coordinate up to scale by gamma
    '''
    x2_hat = fE.getSkew(x2)
    a = np.dot(np.dot(x2_hat, R), x1)
    a = np.reshape(a, (3,1))
    b = np.dot(x2_hat, T)
    M = np.concatenate((a,b), axis=1)
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(M.T, M))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    X = e_vecs[:, np.argmin(e_vals)]     
    lamda = X[0]/X[1]
    return lamda * np.array([x1[0], x1[1], 1]).T

def SelectionXtrans(img1, img2, Rs, Ts):
    kp1,des1 = fE.getDescriptor(img1)
    kp2,des2 = fE.getDescriptor(img2)
    matches = fE.getMatches(des1,des2,count=200)
    F, mask = fE.getFundMat(matches,kp1,kp2)    
    pts1,pts2 = fE.getMatchPts(matches, kp1, kp2, mask)
    N = pts1.shape[0]
    P1 = np.concatenate((pts1, np.ones((N,1))), axis=1)
    P2 = np.concatenate((pts2, np.ones((N,1))), axis=1)
    minus = [0,0]
    for n in range(0,2):
        r, t = Rs[n], Ts[n]
        for i in range(0,N):

            x1 = np.dot(K_inv, P1[i,:].T)
            x2 = np.dot(K_inv, P2[i,:].T)
            X = lstsql3d(r,t, x1, x2)
            if X[2] < 0:
                minus[n] += 1
    print 'minus is ', minus
    if minus[0] > minus[1]:
        return 1
    else:
        return 0

img1 = cv2.imread('imgs/test1.png')
img2 = cv2.imread('imgs/test3.png')
# print 'image shape: ', img1.shape
pts1 = np.array([[184,272],[476,557],[1102,308],[743,128]])
pts2 = np.array([[305,275],[384,545],[1178,361],[889,154]])
kp1,des1 = computeDesc(img1, pts1)
kp2,des2 = computeDesc(img2, pts2)
# lower the threshold to pass the test!!
matches = fE.getMatches(des1,des2,threshold=0.8) #, kp1, kp2, img1, img2)
print len(matches), ' is lenght of matches!'
if (len(matches) < 4):
    raise NameError('all 4 corners are not matched!')
pts1,pts2 = fE.getMatchPts(matches, kp1, kp2)
P1 = np.concatenate((pts1, np.array([[1,1,1,1]]).T), axis=1)
P2 = np.concatenate((pts2, np.array([[1,1,1,1]]).T), axis=1)
# print pts1, pts2
H = fE.getHomography(matches, kp1, kp2)
# normalize
U,S,V = np.linalg.svd(H)
# print S
H = H / S[1]
if (np.dot(P2[0,:], np.dot(H, P1[0,:].T))):
    H = -H
# print H
# A = np.dot(H.T, H)
# U,S,V = np.linalg.svd(A)
# print S
# v1,v2,v3 = V[:,0], V[:,1], V[:,2]
# K is needed because x needs to times K^-1
retval, rotations, translations, normals = cv2.decomposeHomographyMat(H,K)

# H = np.dot(np.dot(K_inv, H), K)
# print np.dot(H - rotations[2], normals[2])
###################################################################
# calculate 3d, "T" is in unit of plane-depth -- d
Rs = []
Ts = []
for i, n in enumerate(normals):
    if (n[2] > 0):  # positive depth constraint
        Rs.append(rotations[i])
        Ts.append(translations[i])
# print Rs, Ts
# no more shit we can do right now, we need more correspondences!
iResult = SelectionXtrans(img1, img2, Rs, Ts)
R = Rs[iResult]
T = Ts[iResult]
print 'The real R is ', R
print 'unit translation length: ', np.linalg.norm(T)

# finally calculate the coordinate of the whole 4 shits
Xs = []
for i in range(0,4):
    x1 = np.dot(K_inv, P1[i,:].T)
    x2 = np.dot(K_inv, P2[i,:].T)
    Xs.append(plane-depth * lstsql3d(R, T, x1, x2))

# print np.linalg.norm(Xs[0] - Xs[1]) 
print Xs


