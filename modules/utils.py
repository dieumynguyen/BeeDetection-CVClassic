import cv2
import matplotlib.pyplot as plt

class VideoLoader:
    def __init__(self, vid_path, color=True, img_limit=None, img_skip=1, start_i=0, end_i=None):
        self.vid_path = vid_path
        self.img_limit = img_limit
        self.img_skip = img_skip
        self.start_i = start_i
        self.end_i = end_i
        self.color_xform = cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY
        
        self._open_stream(vid_path)
        self.num_images_loaded = 0
        
    def _open_stream(self, vid_path):
        self.cap = cv2.VideoCapture(vid_path)
        
    def __iter__(self):
        self.frame_i = 0
        self.num_images_loaded = 0
        return self

    def __next__(self):
        # Read frame and increment frame counter
        ret, frame = self.cap.read()
        self.frame_i += 1
        
        # Check for image limit
        condition_1 = self.img_limit and self.num_images_loaded >= self.img_limit
        condition_2 = self.end_i is not None and self.frame_i >= self.end_i
        if condition_1 or condition_2:
            raise StopIteration
        # Check image skip
        elif (self.frame_i % self.img_skip != 0) or (self.frame_i < self.start_i):
            frame = self.__next__()
        else:
            if frame is None:
                raise StopIteration
            
            frame = cv2.cvtColor(frame, self.color_xform)
            self.num_images_loaded += 1
                
        return frame
    
def imgs2vid(imgs, outpath, fps=12):
    height, width = imgs[0].shape[0:2]
        
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)
    
    for img in imgs:
        video.write(img)
    
    cv2.destroyAllWindows()
    video.release()
    
def get_squarish_rows_cols(num):
    factors = [i for i in range(2, num)[::-1] if num%i==0]
    if len(factors) == 1:
        rows, cols = factors[0], factors[0]
    elif len(factors) == 0:
        rows, cols = num, 1
    else:
        mid = len(factors)//2
        rows = factors[mid]
        cols = factors[mid-1]
    return rows, cols

def plot_cropped_imgs(cropped_imgs, stats):
    nrows, ncols = get_squarish_rows_cols(len(cropped_imgs))
    
    if nrows == 1:
        if ncols > 10:
            fig, ax = plt.subplots(ncols, figsize=(12,12 + 2*ncols))
        else:
            fig, ax = plt.subplots(1, ncols, figsize=(12,12))
        for img_i in range(ncols):
            ax_i = ax[img_i]
            ax_i.imshow(cropped_imgs[img_i], cmap=plt.cm.gray)
            ax_i.set_title(f"Img {img_i}\nArea: {stats[img_i][-1]}")
    elif ncols == 1:
        if nrows > 10:
            fig, ax = plt.subplots(nrows, figsize=(12,12 + 2*nrows))
        else:
            fig, ax = plt.subplots(1, nrows, figsize=(12,12))
        for img_i in range(nrows):
            ax_i = ax[img_i]
            ax_i.imshow(cropped_imgs[img_i], cmap=plt.cm.gray)
            ax_i.set_title(f"Img {img_i}\nArea: {stats[img_i][-1]}")
    else:
        fig, ax = plt.subplots(nrows, ncols, figsize=(12,12))
        img_i = 0
        for row_i in range(nrows):
            for col_i in range(ncols):
                ax_i = ax[row_i][col_i]
                ax_i.imshow(cropped_imgs[img_i], cmap=plt.cm.gray)
                ax_i.set_title(f"Img {img_i}\nArea: {stats[img_i][-1]}")
                img_i += 1
    plt.tight_layout()