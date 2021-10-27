import math

image = imread('rara.jpg')
grad_x = filter(image, 'sobelX')
grad_y = filter(image, 'sobelY')
grad_mag = sqrt( grad_x .^2 + grad_y.^2 )
grad_ori = atan2( grad_y, grad_x )

# Takes in a interest point x,y location and returns a feature descriptor
def SIFTdescriptor(x, y):
    small_box_count = -1
    d = zeros(128,1)
    for row in range(-8, 8,  4):
        for col in range(-8, 8, 4):
            small_box_count =+ 1
            for i in range(0, 3):
                for j in range(0, 3):
                    g = grad_ori[row + i + x][col + j + y]
                    gm = grad_mag[row + i + x][col + j + y]
                    if (g >= -math.pi) & (g > -3*math.pi/4):
                        d[small_box_count*8][1] =+ gm
                    if (g >= -3*math.pi/4) & (g > -math.pi/2):
                        d[small_box_count*8 + 1][1] =+ gm
                    if (g >= -math.pi/2) & (g > -math.pi/4):
                        d[small_box_count*8 + 2][1] =+ gm
                    if (g >= -math.pi/4) & (g > 0):
                        d[small_box_count*8 + 3][1] =+ gm
                    if (g >= 0) & (g > math.pi/4):
                        d[small_box_count*8 + 4][1] =+ gm
                    if (g >= math.pi/4) & (g > math.pi/2):
                        d[small_box_count*8 + 5][1] =+ gm
                    if (g >= math.pi/2) & (g > 3*math.pi/4):
                        d[small_box_count*8 + 6][1] =+ gm
                    if (g >= 3*math.pi/4) & (g > math.pi):
                        d[small_box_count*8 + 7][1] =+ gm
    return d
    