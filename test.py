import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import math as m
from particle import particle
import cv2
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#this code is for the testset. you can change the values as indicated by "#change here to load different dataset"

def setMAP_params(MAP):
    MAP['res'] = 0.05  # meters

    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20

    MAP['xmax'] = 30
    MAP['ymax'] = 30

    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

    MAP['map'] = np.zeros((MAP['sizey'], MAP['sizex']), dtype=np.int32)
    MAP['l_odds'] = np.zeros((MAP['sizey'], MAP['sizex']), dtype=np.float64) 
    return MAP

def setDist_params(dist):
    dist['g2com'] = 0.93
    dist['com2h'] = 0.33
    dist['h2lid'] = 0.15
    dist['g2lid'] = 0.93 + 0.33 + 0.15
    return dist

def plot_poses(poses, t):
    fig = plt.figure(1)
    s1 = fig.add_subplot(311)
    s1.plot(t,poses[:,0], 'r', label="x")
    box = s1.get_position()
    s1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    s2 = fig.add_subplot(312)
    s2.plot(t,poses[:,1], 'g', label="y")
    box = s2.get_position()
    s2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    s3 = fig.add_subplot(313)
    s3.plot(t,poses[:,2], 'b', label="theta")
    box = s3.get_position()
    s3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s3.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def get_laser_points(scan, angles):
    
    ranges = np.double(scan)

    # Extract only those scans and corresponding angles which are within a sensible range
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid].reshape(1, -1)
    angles = angles[indValid].reshape(1, -1)

    # XYZ poistions of the hit points in the 'planar' sensor frame
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    zs = np.zeros(xs.shape)
    ps = np.vstack((xs, ys, zs, np.ones(xs.shape)))

    return ps


# Update weights
def update_w(w_arr, corr):

    # Update weights using logsumexp.
    l = np.log(w_arr) + corr
    l_max = np.amax(l)
    l_sum = l_max + m.log(np.sum(np.exp(l-l_max)))
    l = l - l_sum
    w_up = np.exp(l)

    return w_up

# Stratified resampling using updated particles
def resample(p_list):
    new_list = []
    u = np.random.uniform(0, 1./float(len(p_list)), 1)[0]
    j = 0
    c = p_list[0].w
    for k in xrange(len(p_list)):

        # Divide the circle into equal parts. Move to the next particle only when b > cumulative weight
        b = (u + ((float(k))/(float(len(p_list)))))

        if(b>1):
            print b
            quit()

        while (b > c):
            j = (j+1)
            c = (c + p_list[j].w)

        new_list.append(particle(w = 1./float(len(p_list)),x = p_list[j].x))

    return new_list

# Functions to load and store the trained models
    
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def roundup(x,t_jump):
    return int(m.ceil(x / t_jump)) * t_jump


video_gen = True

t_start = 0
t_jump = 10

lidar_dat = ld.get_lidar('testset/lidar/test_lidar')
#lidar_dat = ld.get_lidar('trainset/lidar/train_lidar0')  #change here to load different dataset
joint_dat = ld.get_joint('testset/joint/test_joint') 
#joint_dat = ld.get_joint('trainset/lidar/train_joint0')  #change here to load different dataset

if video_gen:
    img_dat1 = ld.get_rgb('testset/cam/RGB_1')
    #img_dat1 = ld.get_rgb('trainset/cam/RGB_0')   #change here to load different dataset 
    print 'ok0.1'
    img_dat2 = ld.get_rgb('testset/cam/RGB_2')
    #img_dat2 = ld.get_rgb('trainset/cam/RGB_3_3')
    print 'ok0.2'
    img_dat3 = ld.get_rgb('testset/cam/RGB_3')
    #img_dat3 = ld.get_rgb('trainset/cam/RGB_3_4')
    print 'ok0.3'
    #comment from below while loading different dataset
    img_dat4 = ld.get_rgb('testset/cam/RGB_4')  
    print 'ok0.4'
    img_dat5 = ld.get_rgb('testset/cam/RGB_5')
    print 'ok0.5'
    img_dat6 = ld.get_rgb('testset/cam/RGB_6')
    print 'ok0.6'
    img_dat7 = ld.get_rgb('testset/cam/RGB_7')
    print 'ok0.7'
    img_dat8 = ld.get_rgb('testset/cam/RGB_8')
    print 'ok0.8'
    img_dat9 = ld.get_rgb('testset/cam/RGB_9')
    print 'ok0.9'

print 'ok1'

# Take the time steps of joints which match lidar time steps
joint_ind = []
lidar_t = lidar_dat[0]['t']
for i in xrange(len(lidar_dat)):
    joint_ind.append(np.argmin(np.abs(lidar_dat[i]['t'] - joint_dat['ts'])))
    lidar_t = np.vstack((lidar_t,lidar_dat[i]['t']))

lidar_img_ind = []

if video_gen:
    for i in xrange(len(img_dat1)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat1[i]['t'])),t_jump))
    print 'ok1.1'
    for i in xrange(len(img_dat2)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat2[i]['t'])),t_jump))
    print 'ok1.2'
    for i in xrange(len(img_dat3)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat3[i]['t'])),t_jump))
    print 'ok1.3'
    #comment from below while loading different dataset
    for i in xrange(len(img_dat4)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat4[i]['t'])),t_jump))
    print 'ok1.4'
    for i in xrange(len(img_dat5)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat5[i]['t'])),t_jump))
    print 'ok1.5'
    for i in xrange(len(img_dat6)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat6[i]['t'])),t_jump))
    print 'ok1.6'
    for i in xrange(len(img_dat7)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat7[i]['t'])),t_jump))
    print 'ok1.7'
    for i in xrange(len(img_dat8)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat8[i]['t'])),t_jump))
    print 'ok1.8'
    for i in xrange(len(img_dat9)):
        lidar_img_ind.append(roundup(np.argmin(np.abs(lidar_t - img_dat9[i]['t'])),t_jump))
    print 'ok1.9'

# Pick neck and head angles at only that timestep
neck = joint_dat['head_angles'][0, joint_ind]
head = joint_dat['head_angles'][1, joint_ind]
t = joint_dat['ts'][:, joint_ind]

# # Accumulate the poses for plotting if required
poses = np.array([0.,0.,0.])
for i in xrange(t.shape[1]-1):
    poses = np.vstack((poses,(lidar_dat[i+1]['pose'][0])))

# Initialize MAP
MAP = {}
MAP = setMAP_params(MAP)

# All possible angles of the scanner
angles = np.arange(-135,135.25,0.25)*np.pi/180.

# Distance parameters for the robot
dist = {}
dist = setDist_params(dist)


# Initialize particle list
num_part = 1000
p_list = []
for i in xrange(num_part):
    p = particle(1/float(num_part), lidar_dat[t_start]['pose'])
    p_list.append(p)

# Initialize MAP after getting the coordinates of the laser points from the 1st scan
ps = get_laser_points(lidar_dat[t_start]['scan'][0], angles)
MAP = p_list[0].update_map(MAP, head[t_start], neck[t_start], dist, ps, lidar_dat[t_start]['rpy'][0, 0], lidar_dat[t_start]['rpy'][0, 1])

# Setting threshold for effective number of particles
n_thresh = num_part/3

pred = np.zeros((1,3))

xrob = []
yrob = []

if video_gen:
    print 'ok2'
    video = cv2.VideoWriter('traj.avi', -1, 6, (MAP['sizex'],MAP['sizey']))
    video2 = cv2.VideoWriter('real7.avi', -1, 6, (img_dat3[0]['width'],img_dat3[0]['height']))
    for i in xrange(len(img_dat1)):
        print i
        video2.write(img_dat1[i]['image'])
    for i in xrange(len(img_dat2)):
        print i
        video2.write(img_dat2[i]['image'])
    for i in xrange(len(img_dat3)):
        print i
        video2.write(img_dat3[i]['image'])
    #comment from below while loading different dataset
    for i in xrange(len(img_dat4)):
        print i
        video2.write(img_dat4[i]['image'])
    for i in xrange(len(img_dat5)):
        print i
        video2.write(img_dat5[i]['image'])
    for i in xrange(len(img_dat6)):
        print i
        video2.write(img_dat6[i]['image'])
    for i in xrange(len(img_dat7)):
        print i
        video2.write(img_dat7[i]['image'])
    for i in xrange(len(img_dat8)):
        print i
        video2.write(img_dat8[i]['image'])
    for i in xrange(len(img_dat9)):
        print i
        video2.write(img_dat9[i]['image'])
    video2.release()


    print 'ok3'


for i in xrange(t_start,t.shape[1]-t_jump,t_jump):

    print i

    # noise
    if i>100:
        Q = 1e-4 * np.diag((100., 100., 100.))
    else:
        Q = 0 * np.diag((1., 1., 1.))

    # control action
    delta = (lidar_dat[i+t_jump]['pose']) - (lidar_dat[i]['pose'])
    th = lidar_dat[i]['rpy'][0, 2]

    R_t = np.array([[m.cos(th), m.sin(th)], [-m.sin(th), m.cos(th)]])
    delta[0,0:2] = np.dot(R_t,delta[0,0:2].T).T
    delta[0, 2] = lidar_dat[i + t_jump]['rpy'][0,2] - th
    c_act = delta

    roll = 0
    pitch = 0

    # Noise
    noise = np.random.multivariate_normal(np.zeros((3,)), Q, num_part)


    # Initialize list to store correlations
    corr = []

    # Get local scan points
    ps = get_laser_points(lidar_dat[i+t_jump]['scan'][0], angles)

    # Loop to update the particles and find the correlation
    for j in xrange(len(p_list)):
        
        # Update the particles using control action and noise
        p_list[j].predict(noise[j, :].reshape(1, -1), c_act)

        # Find the correlation
        corr.append(p_list[j].get_corr(MAP, head[i+t_jump], neck[i+t_jump], dist, ps, roll, pitch))

    print max(corr)

    w_arr = np.asarray(list(part.w for part in p_list))
    
    w_up = update_w(w_arr, np.asarray(corr))

    for j in xrange(len(p_list)):
        p_list[j].w = w_up[j]

    # Choose the particle with max weight as the best particle
    best_p = p_list[np.argmax(w_up)]

    # Update the map using the best particle
    MAP = best_p.update_map(MAP, head[i+t_jump], neck[i+t_jump], dist, ps, roll, pitch)
    pred = np.vstack((pred,best_p.x))


    # Check resampling condition and resample using SIR
    n_eff = 1./np.sum(w_up*w_up)
    if n_eff < n_thresh:
        p_list = resample(p_list)

    xrob.append(np.ceil((p_list[0].x[0, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
    yrob.append(np.ceil((-p_list[0].x[0, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)

    img = np.ones((MAP['sizey'], MAP['sizex'],3), dtype=np.uint8) * 127
    mask = MAP['l_odds'] > 10. * m.log(9)
    img[mask,:] = 0
    mask = MAP['l_odds'] < 0
    img[mask,:] = 255
    img[yrob,xrob,:] = 0
    img[yrob, xrob, 2] = 255

    cv2.imshow('img', img)
    cv2.waitKey(1)


    if video_gen:
        if i in lidar_img_ind:
            video.write(img)
            print 'image written'

if video_gen:
    video.release()



img = np.ones((MAP['sizey'], MAP['sizex'], 3), dtype=np.uint8) * 127
mask = MAP['l_odds'] > 0
img[mask, :] = 0
mask = MAP['l_odds'] < 0
img[mask, :] = 255
img[yrob, xrob, :] = 0
img[yrob, xrob, 2] = 255

cv2.imshow('img', img)
cv2.imwrite('res_test.png',img)
cv2.waitKey(0)



fig1 = plt.figure(1)
plt.plot(poses[:,0], poses[:,1])
plt.show()

fig2 = plt.figure(2)
plt.plot(pred[:,0], pred[:,1])
plt.show()
print 'ok'

save_obj(pred,'odom0')
save_obj(MAP, 'map0')
save_obj(img, 'img0')


# Plot the surface.
MAP = load_obj('map0')
ny = MAP['l_odds'].shape[0]
nx = MAP['l_odds'].shape[1]

fig = plt.figure(1)
ax = fig.gca(projection='3d')
x = np.linspace(0, nx-1, nx)
y = np.linspace(0, ny-1, ny)

X, Y = np.meshgrid(x, y)
Z = MAP['l_odds']
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-120, 120)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

