import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
def getDescriptor(img, nfea = 0, cT=0.04):

    #print type(img)
    if (len(img.shape) == 3):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print type(gray)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfea, contrastThreshold=cT)    #nfeatures=1000
    kps,descs = sift.detectAndCompute(gray, None)
    # img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints.jpg',img)
    return (kps,descs)

# def compareDesc(desci, descj):
#   i = rd.randint(0,desci.shape[0] - 1)
#   j = rd.randint(0,descj.shape[0] - 1)
#   # threshold = np.linalg.norm(desci[i,:] - descj[j,:]) / 100
#   # print threshold
#   threshold = 10
#   result = []
#   for i in range(0, desci.shape[0]):
#       for j in range(0, descj.shape[0]):
#           comp = np.linalg.norm(desci[i,:] - descj[j,:])
#           if (comp < threshold):
#               result.add((i,j))
#               return result
#   return result

def getMatches(des1, des2, kp1=None, kp2=None, img1=None, img2=None, count=100, threshold=0.5):
    '''
        @param: descriptor, des1 -- queryIdx,  des2 -- trainIdx
        return: list of tuple as (queryIdx, trainIdx)
    '''
    # kp1, des1 = fE.getDescriptor(fid1,nfea=nfeatures)
    # kp2, des2 = fE.getDescriptor(fid2,nfea=nfeatures)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params,search_params)

    # matches = flann.knnMatch(des1,des2,k=2)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # print "Num of all matches are %d"%len(matches)
    final_matches = []
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    ratio = threshold
    matches.sort(key=lambda m: m[0].distance)
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            matchesMask[i]=[1,0]
            final_matches.append(m)
            if (len(final_matches) > count):
                break
    if (kp1 is None):
        return final_matches

    for m in final_matches:
        print m.distance
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    plt.imshow(img3),plt.show()
    return final_matches

def getHomography(good, kp1, kp2):
    '''
        good: DMatch object list
    '''
    print "The number of matches used in Homography is %d"%len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    ####################################################################################
    # Using RANSAC
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # matchesMask = mask.ravel().tolist()
    # return (M, matchesMask)
    ####################################################################################
    # Using the normal method (instead of RANSAC)
    H, mask = cv2.findHomography(src_pts, dst_pts, 0)     # the four points need to be perfect!
    return H

def getFundMat(good, kp1, kp2):
    '''
        good: DMatch object list
        kp1,kp2: query key points, train key points
    '''
    print 'The number of matches used in fundamental matrix is %d'%len(good)
    pts1,pts2 = getMatchPts(good, kp1, kp2)
    # F, mask = find_fund_mat(pts1,pts2)            #custom version of find fundmat
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask

def drawEpiline(img1, img2, pts1, pts2, F):

    pts1,pts2 = pts1[:30,:], pts2[:30,:]    #scale down a little bit

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)   

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

def getMatchPts(good, kp1, kp2, mask=None):
    '''
        return: np array, N * 2 dim of float32 coordinates
    '''
    pts1,pts2 = [],[]
    for m in good:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    if mask is not None:
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
    return (pts1, pts2)

def drawWarp(img1, M, dsize):
    img_1_2 = cv2.warpPerspective(img1,M,dsize)
    plt.subplot(121),plt.imshow(img1),plt.title('Input')
    plt.subplot(122),plt.imshow(img_1_2),plt.title('Output')
    plt.show()

def getCors(shape, M, origin=False):
    n,m = shape[0], shape[1]
    result = np.zeros((3, n * m),np.float32)
    result[1,:] = np.repeat(np.arange(0,n), m)
    result[0,:] = np.tile(np.arange(0,m), n)
    result[2,:] = np.ones((1,n * m), np.float32)
    if (origin):
        return result[0:2,:]
    result = np.dot(M, result)
    result[0,:] = result[0,:] / result[2,:]
    result[1,:] = result[1,:] / result[2,:]
    xmax = max(result[0,:])
    ymax = max(result[1,:])
    xmin = min(result[0,:])
    ymin = min(result[1,:])
    return (xmax, ymax, xmin, ymin, result[0:2,:])

def getCors2(shape, M, origin=False):
    n,m = shape[0], shape[1]
    result = np.zeros((3, n * m),np.float32)
    result[1,:] = np.repeat(np.arange(0,n), m)
    result[0,:] = np.tile(np.arange(0,m), n)
    result[2,:] = np.ones((1,n * m), np.float32)
    if (origin):
        return result[0:2,:]
    result = np.dot(M, result)
    result[0,:] = result[0,:] / result[2,:]
    result[1,:] = result[1,:] / result[2,:]
    x1 = (result[0,0], result[1,0])
    x2 = (result[0, (n - 1) * m], result[1, (n - 1) * m])
    x3 = (result[0, n * m - 1], result[1, n * m - 1])
    x4 = (result[0, m - 1], result[1, m - 1])
    return (x1, x2, x3, x4, result[0:2,:])

def getmask(img):
    row = img.shape[0]
    col = img.shape[1]
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(row / 3, row * 6 / 7):
        for j in range(col / 3, col * 9 / 10):
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
                mask[i,j] = 1
                print 'hi'

    return mask


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,n = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = (255,0,0)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,5, cv2.LINE_AA)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def getSkew(x):
    return np.array([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0]])

def calcH(pts1, pts2, T, F):
    '''
        @param: pts1, pts2 -- N * 2 coordinates
                T -- 3 colomn vector
                F -- fund mat
        calculate H 
    '''
    N = pts1.shape[0]
    A = np.zeros((N * 3, 3), np.float32)
    b = np.zeros((N * 3, 1), np.float32)
    T_hat = getSkew(T)
    for i in range(0, N):
        x1 = np.ones((3,1), np.float32)
        x2 = np.ones((3,1), np.float32)
        x1[:2] = np.reshape(pts1[i,:], (2,1))
        x2[:2] = np.reshape(pts2[i,:], (2,1))
        hat = getSkew(x2)
        A[3 * i:3 * i + 3,:] = np.dot(np.dot(hat, T), np.transpose(x1))
        b[3 * i:3 * i + 3] = -np.dot(np.dot(hat, np.transpose(T_hat)), np.dot(F, x1))
    v = np.linalg.lstsq(A,b)[0]
    v = np.reshape(v, (3,1))
    H = np.dot(np.transpose(T_hat),F) + np.dot(T, np.transpose(v))
    # print v
    return H

def warpImg(img, H):
    '''
        return: the returned img is fucked by Ht to sit in the top-left corner 
    '''
    h,w = img.shape[:2]
    bd = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    bd_ = cv2.perspectiveTransform(bd, H)
    [xmin, ymin] = np.int32(bd_.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(bd_.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    # print t
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translation matrix
    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return Ht, result

def getDepth(img1, img2, patch_size=10, search_window=200):
    '''
        Patch size is 10 for a 375 * 1290 size image and it's perfect
        
        @param: 
            img1 -- left image, img2 -- right image; same size and grayscale images!!!!
            patch_size -- the patch size to compute correlation
            window: correspondence searching window in uints of pixel

        return: np.uint8 grayscale depth image, disparity of int32
    '''
    h,w = img1.shape[:2]
    p = patch_size
    depth = np.zeros((h,w),np.uint8)
    disparity = np.zeros((h,w),np.int32)
    for i in range(0, h - p):
        for j in range(p, w):
            if (j - search_window < 0):
                window = j
            else:
                window = search_window
            img = img2[i:i + p,j - window:j]        #search along a resonable window
            template = img1[i:i + p,j:j + p]
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # if (i == h / 2):
                # print min_val, max_val, min_loc, max_loc, 'disp: ', window - max_loc[0]
            top_left = max_loc
            disp = window - top_left[0]
            disparity[i,j] = disp
            if (disp != 0):
                depth[i,j] = np.uint8(255 / disp)

    return depth,disparity

def normalize_eight_pts(pts):
    ox, oy = tuple(np.average(pts[:, 0:2], axis=0))
    GT = np.identity(3)
    GT[0, 2] = -ox
    GT[1, 2] = -oy
    dists = np.linalg.norm(pts.dot(GT.T)[:, 0:2], axis=1)
    scale = np.sqrt(2) / np.average(dists)
    GS = np.diag([scale, scale, 1])
    N = GS.dot(GT)
    return N, pts.dot(N.T)

def find_fund_mat_eight_pts(src_pts, dst_pts):
    N1, norm_src_pts = normalize_eight_pts(src_pts)
    N2, norm_dst_pts = normalize_eight_pts(dst_pts)
    
    A = np.zeros([8, 9])
    for i in range(8):
        A[i, :] = np.kron(norm_src_pts[i, :], norm_dst_pts[i, :])
    U, s, V = np.linalg.svd(A)
    vf = (V[8, :] / V[8, 8])
    F = vf.reshape(3, 3).T
    F /= F[2, 2]
    F = N2.T.dot(F).dot(N1)
    F /= F[2, 2]
    return F

def find_fund_mat(src_pts, dst_pts, threshold=1.5): 
    n_pts = src_pts.shape[0]
    src_pts_ext = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1))), axis=1)
    dst_pts_ext = np.concatenate((dst_pts, np.ones((dst_pts.shape[0], 1))), axis=1)
    F = np.zeros((3, 3))
    matchesMask = np.zeros(n_pts)
    
    max_vote = -1
    population = range(n_pts)
    for k in range(2000):
        eight_point_filter = tuple(sorted(random.sample(population, 8)))
        temp_F = find_fund_mat_eight_pts(src_pts_ext[eight_point_filter, :],
                                         dst_pts_ext[eight_point_filter, :])
        if temp_F is not None:
            epilines = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, temp_F).reshape(-1, 3)
            dists = np.abs(np.sum(epilines * dst_pts_ext, axis=1))
            temp_mask = dists <= threshold
            if np.sum(temp_mask) > max_vote:
                max_vote = np.sum(temp_mask)
                F = temp_F
                matchesMask = temp_mask
            
    return F, matchesMask
