import numpy as np
import math as m
import cv2

class particle(object):

    # Particle has weight w and the current state x
    def __init__(self, w,x):
        self.w = w
        self.x = x
        
    # Update the map by finding hit coordinates in the world frame using the head/neck angles, local frame
    # coordinates and robot distance coordinates
    def update_map(self, MAP, head, neck, dist, ps, roll, pitch):

        # Transform points from the world frame to the local frame
        pw = self.loc2wld(head, neck, dist, ps, roll, pitch)

        # Check if the ray is striking the ground plane using the world Z coordinate
        indValid = (pw[2, :] > 0.1)
        pw = pw[:, indValid]

        # Convert from physical to map coordinates
        # xs remain in the same scale. For ys the axis needs to be inverted and distances measured from ymin.
        xis = np.ceil((pw[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yis = np.ceil((-pw[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        # Position of the robot from physical to map coordinates
        xrob = np.ceil((self.x[0, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yrob = np.ceil((-self.x[0, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        # Pick only those coordinates which are within the image limits
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)),
                                                (xis < MAP['sizex'])), (yis < MAP['sizey']))
        xis = xis[indGood]
        yis = yis[indGood]

        # Make a contour of the hit points that begins and ends with the robot pose
        cnt = np.vstack((xis, yis)).T
        cnt = np.vstack((np.array([[xrob, yrob]]), cnt, np.array([[xrob, yrob]])))

        # Mask the scan region using the hit cells as polygon vertices and fill with log(1/9)
        mask = np.zeros(MAP['map'].shape, dtype=np.float64)
        cv2.drawContours(image=mask, contours=[cnt], contourIdx=0, color=(0.5 * m.log(1./9.)), thickness=-1)


        # add mask to log-odds map to accumulate log(1/9) in the free region
        MAP['l_odds'] += mask

        # Add (2 * log 9) to the points where it is hit (since it is set as log(1/9) during the draw contours function)
        MAP['l_odds'][yis, xis] += (2 * m.log(9.))

        # Clip the function so that it does not get too confident about a particular reading
        MAP['l_odds'] = np.clip(MAP['l_odds'], -100, 100)

        # Find values where the log odds is greater than 0 (probability more than half) and set the map values there as
        # 1. Note reinitialize MAP before setting the values. Never reset the log odds
        mask = MAP['l_odds'] > 10. * m.log(9)

        # Set the points where log odds is greater than 0 as 1
        MAP['map'] = np.zeros((MAP['sizey'], MAP['sizex']), dtype=np.uint8)  # DATA TYPE: char or int8
        if np.any(mask):
            MAP['map'][mask] = MAP['l_odds'][mask] * 100 / np.amax(MAP['l_odds'][mask])

        return MAP

    # Convert the points from the local frame of the scanner to the world frame
    def loc2wld(self, head, neck, dist, ps, roll, pitch):

        # Robot pose
        x = self.x[0, 0]
        y = self.x[0, 1]
        th = self.x[0, 2]

        # Transformation from robot base to world
        Tr2w = np.dot(np.array([[m.cos(th), -m.sin(th), 0, x],
                         [m.sin(th), m.cos(th), 0, y],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]]),
                      np.dot(np.array([[m.cos(pitch), 0, m.sin(pitch), 0],
                                       [0, 1., 0, 0],
                                       [-m.sin(pitch), 0., m.cos(pitch), 0.],
                                       [0., 0., 0., 1.]]),
                             np.array([[1., 0, 0, 0],
                                       [0, m.cos(roll), -m.sin(roll), 0],
                                       [0., m.sin(roll), m.cos(roll), 0.],
                                       [0., 0., 0., 1.]])))

        # Transformation from the robot head to the base
        Th2r = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., dist['g2com'] + dist['com2h']],
                         [0., 0., 0., 1.]])

        # Transformation from lidar to head = R_yaw * R_pitch * R_trans
        Tl2h = np.dot(np.dot(np.array([[m.cos(neck), -m.sin(neck), 0., 0.],
                                       [m.sin(neck), m.cos(neck), 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]]),
                             np.array([[m.cos(head), 0., m.sin(head), 0.],
                                       [0., 1., 0., 0.],
                                       [-m.sin(head), 0., m.cos(head), 0.],
                                       [0., 0., 0., 1.]])),
                      np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., dist['h2lid']],
                                [0., 0., 0., 1.]]))

        # Transform from  local coordinates to global coordinates
        pw = np.dot(Tr2w, np.dot(Th2r, np.dot(Tl2h, ps)))
        return pw

    def predict(self, noise, c_act):

        # print hex(id(self.x))
        # Use current theta
        th = self.x[0,2]

        # Find rotation matrix to convert from local to global
        R = np.array([[m.cos(th), -m.sin(th)], [m.sin(th), m.cos(th)]])

        a = np.zeros((1,3))

        # Convert from local to global frame and add the noise
        a[0, 0:2] = self.x[0,0:2] + (np.dot(R,c_act[0, 0:2].T).T + noise[0, 0:2])
        a[0, 2] = self.x[0,2] + (noise[0, 2] + c_act[0, 2])
        self.x = a


    def get_corr(self, MAP, head, neck, dist, ps, roll, pitch):


        # Transform points from the world frame to the local frame
        pw = self.loc2wld(head, neck, dist, ps, roll, pitch)

        # Check if the ray is striking the ground plane
        indValid = (pw[2, :] > 0.1)
        pw = pw[:, indValid]

        # Convert from physical to map coordinates
        xis = np.ceil((pw[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        yis = np.ceil((-pw[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        # Pick only those coordinates which are within the image limits
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)),
                                                (xis < MAP['sizex'] - 1 )), (yis < MAP['sizey'] -1))


        xis = xis[indGood]
        yis = yis[indGood]
        
        cmax = 0.
        xreq = 0.
        yreq = 0.
        
        x_range = np.arange(-2, 3, 1)
        y_range = np.arange(-2, 3, 1)
        
        for i in xrange(x_range.shape[0]):
        
            for j in xrange(y_range.shape[0]):
                c = np.sum(MAP['map'][yis+y_range[j],xis + x_range[i]])/100
             
                if c > cmax:
                    cmax = c
                    xreq = float(x_range[i]) * MAP['res']
                    yreq = -float(y_range[j]) * MAP['res']
        
        x_cur = self.x[0,0] + xreq
        y_cur = self.x[0,1] + yreq
        
        a = np.array([[x_cur, y_cur, self.x[0,2]]])
        self.x = a
        
        return cmax
