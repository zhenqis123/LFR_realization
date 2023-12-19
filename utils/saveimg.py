import cv2
import numpy as np
def saveimg(image, path):
    if(image.type() == 'torch.FloatTensor'):
        image = image.numpy()
        image = image*255
        image = image.astype(np.uint8)
        image = image.transpose(1, 2, 0)
    elif(image.type() == 'torch.cuda.FloatTensor'):
        image = image.cpu().numpy()
        image = image*255
        image = image.astype(np.uint8)
        image = image.transpose(1, 2, 0)
    elif(image.type() == 'torch.cuda.ByteTensor'):
        image = image.cpu().numpy()
        image = image.transpose(1, 2, 0)
    elif(image.type() == 'torch.ByteTensor'):
        image = image.numpy()
        image = image.transpose(1, 2, 0)
    elif(image.type() == 'numpy.ndarray'):
        if(image.dtype == np.float32):
            image = image.astype(np.uint8)
        if(image.shape[0] == 1):
            image = image.squeeze(0)
    cv2.imwrite(path, image)